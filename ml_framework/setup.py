from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "ml.tensor_ext",
        sources = ["ml/tensor_ext/binder.cpp",
                   "ml/tensor_ext/tensor_base.cpp", ],
    ),
]

setup(
    name="ml",
    version="0.0.1",
    packages=["ml", 
              "ml.nn"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)