"""``python3 -m tolly_router`` runs the HTTP service.

Convenience wrapper so the systemd unit can use the module form.
"""
from .service import main

if __name__ == "__main__":
    main()
