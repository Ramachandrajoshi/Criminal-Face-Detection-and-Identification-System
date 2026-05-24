"""
Integration test: audit_log immutability.
"""

import os
import pytest


class TestAuditImmutability:
    """Verify audit_log is append-only (no UPDATE/DELETE)."""

    def test_audit_log_sql_has_no_delete_grant(self):
        """Verify init.sql does not contain DELETE/UPDATE on audit_log."""
        # Robustly search upwards for the db/init.sql file to support both host and container layouts
        current_dir = os.path.dirname(os.path.abspath(__file__))
        init_sql = None
        while current_dir and current_dir != os.path.dirname(current_dir):
            candidate = os.path.join(current_dir, "db", "init.sql")
            if os.path.exists(candidate):
                init_sql = candidate
                break
            current_dir = os.path.dirname(current_dir)

        if not init_sql:
            pytest.fail("Could not locate db/init.sql in parent directories")

        with open(init_sql) as f:
            content = f.read().upper()

        # audit_log table must not have UPDATE or DELETE in grants
        assert "ALTER TABLE AUDIT_LOG" not in content
        assert "UPDATE AUDIT_LOG" not in content
        assert "DELETE FROM AUDIT_LOG" not in content

    def test_audit_route_is_read_only(self):
        """Verify the audit route only defines GET."""
        from fastapi import APIRouter
        from app.api.routes.audit import router

        # Check that only GET is registered
        routes = [r.methods for r in router.routes if hasattr(r, 'methods')]
        for route_methods in routes:
            for method in route_methods:
                assert method == "GET", f"Audit route should only support GET, found: {method}"

    def test_no_audit_delete_endpoint(self):
        """There should be no DELETE endpoint for audit_log."""
        from app.main import app
        delete_routes = [
            r for r in app.routes
            if hasattr(r, 'methods') and "DELETE" in r.methods
        ]
        for route in delete_routes:
            path = route.path if hasattr(route, 'path') else str(route)
            assert "audit" not in path.lower(), f"DELETE endpoint found on {path}"

    def test_no_audit_update_endpoint(self):
        """There should be no PUT/PATCH endpoint for audit_log."""
        from app.main import app
        for method in ("PUT", "PATCH"):
            update_routes = [
                r for r in app.routes
                if hasattr(r, 'methods') and method in r.methods
            ]
            for route in update_routes:
                path = route.path if hasattr(route, 'path') else str(route)
                assert "audit" not in path.lower(), \
                    f"{method} endpoint found on {path}"
