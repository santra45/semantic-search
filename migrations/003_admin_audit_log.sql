-- 003 — admin_audit_log: who changed what, and whether the cache heard about it
--
-- ADMIN_CONSOLE_PLAN.md §5.2. Append-only by convention: no UPDATE or DELETE
-- path is exposed anywhere in the API. Nothing enforces that at the database
-- level on purpose — a trigger or a REVOKE would also block the one legitimate
-- correction (a migration fixing a malformed row) and would be worked around in
-- a hurry the day it mattered.
--
-- Every write in the admin API goes through one mutate() wrapper that inserts
-- here in the SAME TRANSACTION as the mutation it describes. If the audit
-- insert fails the mutation rolls back: an unlogged disable is worse than a
-- failed one.

CREATE TABLE IF NOT EXISTS `admin_audit_log` (
  `id`            BIGINT       NOT NULL AUTO_INCREMENT,

  -- NULL for break-glass actions taken with X-Operator-Key before real accounts
  -- exist. NOT a foreign key, and not by omission: an audit row must outlive
  -- the account that made it, and ON DELETE SET NULL would erase which of two
  -- deleted operators did the thing.
  `admin_user_id` VARCHAR(36)  NULL DEFAULT NULL,
  `actor_email`   VARCHAR(255) NOT NULL
                  COMMENT 'Denormalised so the trail survives a deleted admin_user. "break-glass" for operator-key actions.',

  `action`        VARCHAR(64)  NOT NULL
                  COMMENT 'Dotted verb: client.disable, subscription.pause, licence.revoke, site.environment.',
  `target_type`   VARCHAR(32)  NOT NULL
                  COMMENT 'client | site | subscription | licence | product | admin_user',
  `target_id`     VARCHAR(64)  NOT NULL,

  -- Snapshots, not diffs. A diff needs the reader to already know the shape of
  -- the row; before/after answers "what did this look like" years later when
  -- the columns have moved on. before_json is also what the console's 24h
  -- one-click revert replays, so it has to be complete enough to restore from.
  `before_json`   JSON         NULL DEFAULT NULL,
  `after_json`    JSON         NULL DEFAULT NULL,

  `reason`        VARCHAR(500) NULL DEFAULT NULL
                  COMMENT 'Mandatory at the API layer for anything that takes something offline; NULL-able here because reads and non-destructive edits do not carry one.',

  -- HOW MANY CACHE KEYS THIS WRITE ACTUALLY FORGOT.
  --
  -- auth_cache holds a resolved licence context for 300 seconds and eviction is
  -- deliberately the caller's job, so a mutation that skips it returns 200,
  -- shows success, and changes nothing for five minutes. That failure is
  -- invisible from the outside and gets tested by someone who waits, sees it
  -- work, and concludes it works.
  --
  -- Recording the count makes it visible after the fact: a disable with
  -- evicted = 0 on a subscription that had a live key is the bug, sitting in a
  -- queryable column instead of in nobody's memory. NULL means the action had
  -- no cache dimension at all (creating an admin user); 0 means it had one and
  -- evicted nothing.
  `evicted`       INT          NULL DEFAULT NULL,

  `ip`            VARCHAR(45)  NULL DEFAULT NULL,
  `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (`id`),
  -- "What has ever happened to this licence" — the per-target timeline.
  KEY `idx_audit_target` (`target_type`, `target_id`),
  -- "What happened last Tuesday" — the console's default view, newest first.
  KEY `idx_audit_created` (`created_at`),
  -- "What has this operator been doing" — the question asked when something is
  -- wrong and nobody is admitting to it.
  KEY `idx_audit_actor` (`actor_email`, `created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
