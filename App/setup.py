import sys
from cx_Freeze import setup, Executable

base = 'Win32GUI' if sys.platform=='win32' else None

executables = [
    Executable('main.py', base=base)
]


setup(
    name="Subtitle Generator",
    version="1.0.1",
    description="Generating Subtitle from .mp3 file",
    options={
        "build_exe":{
            "no_compress": True,
        }
    },
    executables=[Executable("main.py")],
)