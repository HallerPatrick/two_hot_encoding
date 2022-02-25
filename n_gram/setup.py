from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="ngram",
    ext_modules=[cpp_extension.CppExtension(name="ngram", sources=["ngram.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

