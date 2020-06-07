from setuptools import setup

if __name__ == "__main__":
    setup(name='stsearch',
          version='0.0.1',
          description='Content-based Spatial-Temporal Search in Video',
          url='https://github.com/fzqneo/stsearch',
          author='Ziqiang Feng',
          author_email='zf@cs.cmu.edu',
          license='Apache 2.0',
          packages=['stsearch'],
          install_requires=['rekallpy', 'numpy', 'opencv-python', 'python-constraint', 'tqdm', 'cloudpickle',
                            'urllib3', 'requests'],
          setup_requires=['pytest-runner'],
          tests_require=['pytest'],
          zip_safe=False)
