"""
RAPIDS Version Information
"""

__version__ = "1.1.0"
__version_info__ = (1, 1, 0)

# Version history
# 1.1.0 - Added interactive 3D visualization with energy display
# 1.0.0 - Initial release with core functionality

def get_version():
    """Return the current version string"""
    return __version__

def get_version_info():
    """Return version as tuple (major, minor, patch)"""
    return __version_info__