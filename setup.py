import os
import sys
import glob

from importlib import import_module
from setuptools import Extension
from setuptools.command.build_ext import build_ext

# -------- quicklib direct/bundled import, copy pasted --------------------------------------------
is_packaging = not os.path.exists("PKG-INFO")
if is_packaging:
    import quicklib
else:
    zips = glob.glob("quicklib_incorporated.*.zip")
    if len(zips) != 1:
        raise Exception("expected exactly one incorporated quicklib zip but found %s" % (zips,))
    sys.path.insert(0, zips[0])
    import quicklib
    sys.path.pop(0)
# -------------------------------------------------------------------------------------------------


class PostponedIncludeGetter:
    def __init__(self, module_name):
        self.module_name = module_name

    def __str__(self):
        mod = import_module(self.module_name)
        return mod.get_include()


leven_root = 'epic/sklearn/metrics/leven'
leven_ext = Extension(
    name=leven_root.replace('/', '.'),
    sources=[f'{leven_root}/leven.cpp'],
    include_dirs=[
        leven_root,
        PostponedIncludeGetter('leven'),
        PostponedIncludeGetter('numpy'),
        PostponedIncludeGetter('pybind11'),
    ],
    language='c++',
)


class BuildExtWithCompilerFlags(build_ext):
    def build_extension(self, ext):
        if ext == leven_ext:
            compile_flags = []
            link_flags = []
            ctype = self.compiler.compiler_type
            if ctype == 'unix':
                compile_flags += [
                    '-O3',
                    '-march=native',
                    f'-DVERSION_INFO="{self.distribution.get_version()}"',  # not important!
                    '-std=c++2a',
                    '-fvisibility=hidden',
                ]
                if sys.platform == 'darwin':
                    mac_flags = ('-stdlib=libc++', '-mmacosx-version-min=10.14')
                    for flags in (compile_flags, link_flags):
                        flags.extend(mac_flags)
                else:
                    compile_flags.append("-fopenmp")
                    link_flags += ['-fopenmp', '-pthread']
            elif ctype == 'msvc':
                compile_flags += [
                    '/EHsc',
                    '/openmp',
                    '/O2',
                    '/std:c++20',
                    fr'/DVERSION_INFO=\"{self.distribution.get_version()}\"',
                ]
            ext.extra_compile_args.extend(compile_flags)
            ext.extra_link_args.extend(link_flags)
        super().build_extension(ext)


quicklib.setup(
    name='epic-sklearn',
    description='An expansion pack for scikit-learn',
    long_description=dict(
        filename='README.md',
        content_type='text/markdown',
    ),
    author='Assaf Ben-David, Yonatan Perry, Uri Sternfeld',
    license='MIT License',
    url='https://github.com/Cybereason/epic-sklearn',
    python_requires=">=3.10",
    top_packages=['epic'],
    version_module_paths=['epic/sklearn'],
    ext_modules=[leven_ext],
    cmdclass={'build_ext': BuildExtWithCompilerFlags},
    zip_safe=False,
    setup_requires=[
        'numpy>=1.21.5',
        'pybind11>=2.7.0',
        'leven @ git+https://github.com/bd-assaf/leven',
    ],
    install_requires=[
        'numpy>=1.21.5',
        'pandas>=1.4.4',
        'scipy>=1.7.3',
        'scikit-learn>=1.1.1',
        'cytoolz',
        'matplotlib',
        'joblib',
        'ultima',
        'epic-logging',
        'epic-common',
        'epic-pandas',
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
    ],
)
