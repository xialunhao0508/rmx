from setuptools import setup, find_packages

setup(
    name="rmx",
    version="0.1.0",
    description='该模型可以根据目标物体在第一帧的掩码(mask)描述，在后续帧中跟踪该物体并生成相应的掩码(mask)。',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="http://192.168.0.188:8090/ai_lab_rd02/ai_sdks/xmem.git",
    packages=find_packages(exclude=['tests']),
    install_requires=["numpy", "torchvision", "pillow"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    zip_safe=False,
)
