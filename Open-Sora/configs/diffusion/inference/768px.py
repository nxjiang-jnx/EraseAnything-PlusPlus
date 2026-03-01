_base_ = [  # inherit grammer from mmengine
    "256px.py",
    "plugins/tp.py",  # use sequence parallel，原来是sp现在是tm能运行
]

sampling_option = dict(
    resolution="768px",
)
