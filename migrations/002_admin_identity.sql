-- 002 — admin identity: who can log into the operator console
--
-- ADMIN_CONSOLE_PLAN.md §5.1. These are OPERATOR accounts — Czargroup staff —
-- and have nothing to do with `clients`, which are merchants. Nothing here ever
-- authenticates a storefront request; that is licences + request_auth.
--
-- Idempotent: CREATE TABLE IF NOT EXISTS throughout, no data seeded. The first
-- owner account is created by the Phase 2 bootstrap CLI, not by a migration —
-- a password hash in a .sql file in git is a credential in git.

CREATE TABLE IF NOT EXISTS `admin_users` (
  `id`            VARCHAR(36)  NOT NULL,
  `email`         VARCHAR(255) NOT NULL,
  `name`          VARCHAR(255) NOT NULL,
  `password_hash` VARCHAR(255) NOT NULL
                  COMMENT 'passlib CryptContext output, algorithm-tagged. Never a bare digest.',
  `role`          ENUM('viewer','operator','owner') NOT NULL DEFAULT 'viewer'
                  COMMENT 'Ordered: viewer < operator < owner. Least privilege by default - a row created without a role reads everything and changes nothing.',
  `is_active`     TINYINT(1)   NOT NULL DEFAULT 1,
  `last_login_at` DATETIME     NULL DEFAULT NULL,
  `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  -- Login looks a user up by email on every attempt, and two accounts sharing
  -- one address is an ambiguity the auth path has no defined answer for.
  UNIQUE KEY `uq_admin_users_email` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE IF NOT EXISTS `admin_sessions` (
  -- NOT the cookie value. This is sha256(token), and the distinction is the
  -- entire security property of the table: the plaintext token exists only in
  -- the operator's browser, so a dump of this table hands over no live session.
  -- Storing the token itself would make read access to MySQL equivalent to
  -- being logged in as an owner. (Contrast `licences`, which now deliberately
  -- does store plaintext — different trade, different threat, and the reasoning
  -- is written out above LICENCES_TABLE in schema_v2.py.)
  `id`            VARCHAR(64)  CHARACTER SET ascii COLLATE ascii_bin NOT NULL
                  COMMENT 'SHA-256 hex of the session cookie token. Never the token.',
  `admin_user_id` VARCHAR(36)  NOT NULL,
  `ip`            VARCHAR(45)  NULL DEFAULT NULL
                  COMMENT '45 chars: an IPv6 address with an IPv4 tail is 45.',
  `user_agent`    VARCHAR(255) NULL DEFAULT NULL,
  `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `expires_at`    DATETIME     NOT NULL,
  `revoked_at`    DATETIME     NULL DEFAULT NULL
                  COMMENT 'Set on logout and on password change. A session is dead if this is set OR expires_at has passed - check both.',
  PRIMARY KEY (`id`),
  KEY `idx_admin_sessions_user` (`admin_user_id`),
  -- Expiry sweeps scan on this; without it they table-scan a table that only
  -- ever grows.
  KEY `idx_admin_sessions_expires` (`expires_at`),
  -- ON DELETE CASCADE: deleting an operator must not leave their live sessions
  -- authenticating. The audit trail does not rely on this table surviving -
  -- admin_audit_log denormalises actor_email for exactly that reason.
  CONSTRAINT `fk_admin_sessions_user`
    FOREIGN KEY (`admin_user_id`) REFERENCES `admin_users` (`id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
