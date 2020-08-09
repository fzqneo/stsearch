"""This module defines classes that conform to stsearch.op.Op.
It intends to wrap legacy Diamond filters as a STSearch Op.
"""
import socket
import time
from typing import Iterable

import PIL.Image
import yaml

import opendiamond.attributes
import opendiamond.client.search
import opendiamond.filter
import opendiamond.server.filter
import opendiamond.server.object_
import opendiamond.server.statistics

from stsearch.op import Graph, Filter
from stsearch.videolib import ImageInterval

FILTER_PORT = 5555

class RGBDiamondSearch(Graph):
    """Wraps a Diamond filter that examines an object's `.rgbimage`.
    
    Args:
        Graph ([type]): [description]
    """

    def __init__(
        self,
        diamond_session: opendiamond.filter.Session,
        filterspecs: Iterable[opendiamond.client.search.FilterSpec]
    ):
        self.session = diamond_session
        self.filterspecs = filterspecs

    def call(self, instream):
        # pass None as state:SearchState into runner. Operations that calls the state
        # will fail, e.g., update-session-vars, state.context.ensure_resource
        runners = [
            opendiamond.server.filter._FilterRunner(None, _DuckFilter(fs, self.session)) 
            for fs in self.filterspecs]
        self.runners = runners

        # pred_fn (query time)
        # create dummy Diamond object with RGB value
        # evaluate with runners
        # pass or drop
        def pred_fn(intrvl: ImageInterval) -> bool:
            diamond_obj = opendiamond.server.object_.Object('no-server-id', 'no-url', compute_signature=False)
            diamond_obj[opendiamond.server.object_.ATTR_DATA] = intrvl.jpeg
            diamond_obj['_rgb_image.rgbimage'] = \
                opendiamond.attributes.RGBImageAttributeCodec().encode(PIL.Image.fromarray(intrvl.rgb))

            return True

class _DuckFilter(object):
    """Duck type ``opendiamond.server.filter.Filter`` to provide only
    necessary members needed by ``opendiamond.server.filter._FilterRunner`.

    """

    def __init__(self, filterspec:opendiamond.client.search.FilterSpec, session:opendiamond.filter.Session):
        self.filterspec = filterspec
        self.session = session

        self.stats = opendiamond.server.statistics.FilterStatistics(filterspec.name)
        self.name = self.signature = filterspec.name
        self.min_score = filterspec.min_score
        self.max_score = filterspec.max_score

    def connect(self) -> opendiamond.server.filter._FilterConnection:
        # currently only support Python filters that supports --tcp flag
        C = yaml.full_load(self.filterspec.code.data)
        docker_image = C['docker_image']
        filter_command = C['filter_command']
        docker_port = FILTER_PORT
        docker_command = f"{filter_command} --filter --tcp --port {docker_port}"

        # 1. Get container handler
        h = self.session.ensure_resource(
            scope='session',
            rtype='docker',
            params=[docker_image, docker_command]
        )
    
        # 2. create connection to filter
        host, port = h['IPAddress'], docker_port
        for _ in range(10):
            try:
                # OS may give up with its own timeout regardless of timeout here
                sock = socket.create_connection((host, port), 1.0)
                break
            except socket.error:
                sock = None
                time.sleep(0.5)
                continue

        if sock is None:
            raise opendiamond.server.filter.FilterExecutionError('Unable to connect to filter at %s: %d' % (host, port))

        return opendiamond.server.filter._FilterTCP(
            sock=sock,
            name=self.name,
            args=self.filterspec.arguments,
            blob=self.filterspec.blob_argument.data)
