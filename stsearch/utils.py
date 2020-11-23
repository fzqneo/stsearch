from stsearch.interval_stream import IntervalStream

def run_to_finish(output_stream):
    output_sub = output_stream.subscribe()
    output_stream.start_thread_recursive()
    return list(iter(output_sub))