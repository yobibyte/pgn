from distutils.core import setup

setup(name='pgn',
      version='0.2',
      description='Pytorch graph networks',
      author='Vitaly Kurin',
      author_email='vitaliykurin@gmail.com',
      packages=['pgn'],
      requires=['torch', 'numpy', 'networkx', 'torch_scatter', ]
     )
