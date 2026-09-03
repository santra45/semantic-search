"""The operator admin console: identity, RBAC, audit, and the write wrapper.

ADMIN_CONSOLE_PLAN.md §6. Everything in this package is about CZARGROUP STAFF
logging into an internal console. None of it authenticates a storefront request
— that is licences plus services/request_auth.authorize_request(), and the two
must never be confused: an admin_user is an operator, a client is a merchant.

Import discipline: this package imports from services/, never the reverse. A
service that reaches back into admin/ would make the console load-bearing for
serving traffic, which it is emphatically not — the API must run identically
with this package deleted.
"""
