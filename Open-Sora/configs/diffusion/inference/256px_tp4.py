_base_ = [  # inherit grammer from mmengine
    "256px.py",
    "plugins/tp4.py",  # use tensor parallel with 4 GPUs
]
