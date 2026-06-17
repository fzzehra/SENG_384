"""Windows non-ASCII path workaround for MediaPipe.

MediaPipe's C++ resource loader cannot open its model files (.binarypb / .tflite)
when the install path contains non-ASCII characters (e.g. the Turkish dotless
"ı" in a Windows username like "Aslı Ceren Can"). The files exist on disk but
loading fails with FileNotFoundError.

This shim rewrites the absolute path MediaPipe builds for its own resources to
the equivalent Windows 8.3 short path, which is always ASCII. It only touches
paths that contain "mediapipe" and are non-ASCII, so the rest of the app is
unaffected. Call apply() once at startup, before any FaceMesh is created.
"""

import os

_applied = False


def apply():
    global _applied

    if _applied or os.name != "nt":
        return

    import ctypes

    original_abspath = os.path.abspath

    def abspath_short(path):
        resolved = original_abspath(path)
        try:
            if (
                "mediapipe" in resolved.lower()
                and not resolved.isascii()
                and os.path.exists(resolved)
            ):
                buffer = ctypes.create_unicode_buffer(1024)
                if ctypes.windll.kernel32.GetShortPathNameW(resolved, buffer, 1024):
                    return buffer.value
        except Exception:
            pass
        return resolved

    os.path.abspath = abspath_short
    _applied = True
    print("WIN PATH FIX: MediaPipe non-ASCII path workaround active")
