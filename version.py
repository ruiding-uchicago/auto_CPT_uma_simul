"""
RAPIDS Version Information
"""

__version__ = "1.4.0"
__version_info__ = (1, 4, 0)

# Version history
# 1.4.0 - Relative positioning, contact distance mode, automatic solvation
# 1.3.0 - Multi-target batch screening, energy definitions in UI/docs
# 1.2.1 - Fixed batch screening functionality with complete visualization suite
# 1.2.0 - Added Web GUI interface with real-time streaming
# 1.1.0 - Added interactive 3D visualization with energy display
# 1.0.0 - Initial release with core functionality

def get_version():
    """Return the current version string"""
    return __version__

def get_version_info():
    """Return version as tuple (major, minor, patch)"""
    return __version_info__
