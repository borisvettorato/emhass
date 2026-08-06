from datetime import datetime, timedelta
try:
    from datetime import UTC
except ImportError:
    # Python 3.10 compatibility
    from datetime import timezone
    UTC = timezone.utc