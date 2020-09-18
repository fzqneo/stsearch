# https://github.com/johannestang/yolo_service/blob/master/examples/examples_detect.py
# https://github.com/johannestang/yolo_service

import io
import time

import flask
import requests
from PIL import Image

app = flask.Flask(__name__)

WRAPPED_HOST = 'http://localhost:8080'
DEFAULT_THRESH = 0.01

# for profileing
avg_time_per_image = 0.

@app.route("/", methods=["GET", "POST"])
@app.route("/detect", methods=["POST"])
def detect():

    if flask.request.method == "POST":
        rv = {"success": False}

        # just read the first file
        _, f = next(iter(flask.request.files.items()))
        content = f.read()

        # get image H and W
        img = Image.open(io.BytesIO(content))
        W, H = img.width, img.height

        try:
            tic = time.time()

            r = requests.post(
                WRAPPED_HOST + '/detect',
                files = {'image_file': io.BytesIO(content)},
                data = {'threshold': DEFAULT_THRESH}
            )

            assert r.ok, "reponse code: " + r.status_code

            # Format from wrapped service: list of (class_name, confidence*100, [x_center, y_center, width, height])
            # box coords are pixels

            # convert to TF format. we don't have detection_classes id here. Hope the client doesn't need it.
            # TF boxes are relative: top, left, bottom, right
            dets = r.json()
            dets = sorted(dets, key=lambda x: x[1], reverse=True)
            rv['num_detections'] = len(dets)
            rv['detection_names'] = [ d[0] for d in dets]
            rv['detection_scores'] = [ d[1]/100. for d in dets]
            rv['detection_boxes'] = []
            for d in dets:
                xc, yc, w, h = d[2]
                x1, x2, y1, y2 = xc - w/2, xc + w/2, yc - h/2, yc + h/2
                left, right, top, bottom = x1/W, x2/W, y1/H, y2/H # pixel to relative
                rv['detection_boxes'].append([top, left, bottom, right])

            rv['success'] = True

            elapsed = time.time() - tic
            global avg_time_per_image
            avg_time_per_image = .5 * avg_time_per_image + .5 * elapsed
            print("cumul avg {:.0f} ms/image".format(1000*avg_time_per_image))
            
        except Exception as e:
            rv['success'] = False
            rv['status_code'] = r.status_code
            rv['exception'] = str(e)
        finally:
            return flask.jsonify(rv)

    else:
        # If GET, show the file upload form:
        return '''
        <!doctype html>
        <title>Object Detection Web API</title>
        <h1>Upload a picture for object detection</h1>
        <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
        </form>
        '''

if __name__ == "__main__":
    print("Running wrapper app")
    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True
    )