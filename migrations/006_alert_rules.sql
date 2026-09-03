-- 006 — alert_rules: thresholds a scheduled job evaluates
--
-- ADMIN_CONSOLE_PLAN.md §5.5. Surfaced in-console first, email later. No job
-- reads this yet — the table lands in Phase 1 so Phase 8 is configuration
-- rather than a migration on a live database.
--
-- `catalogue_pct` is the metric most likely to fire first and is the one the
-- original plan did not have. It is the only ceiling actually enforced today:
-- sites.indexed_items against sites.catalogue_limit blocks writes for real,
-- whereas `quota_pct` is observational until AICHATBOT_QUOTA_ENFORCEMENT is
-- armed — it is not set in the production .env at all. An alert on a limit
-- nothing enforces is a reminder, not a warning, and the console must label it
-- as such.

CREATE TABLE IF NOT EXISTS `alert_rules` (
  `id`            VARCHAR(36)    NOT NULL,

  -- Three scopes because v2 has three levels that mean different things: the
  -- whole customer, one of their store installs, or (via the site) one module.
  -- A rule scoped 'global' has both ids NULL and applies to everything.
  `scope`         ENUM('global','client','site') NOT NULL DEFAULT 'global',
  `client_id`     VARCHAR(36)    NULL DEFAULT NULL,
  `site_id`       VARCHAR(36)    NULL DEFAULT NULL,

  `metric`        VARCHAR(32)    NOT NULL
                  COMMENT 'monthly_cost | quota_pct | catalogue_pct | error_rate | licence_expiry_days',

  -- DECIMAL, not FLOAT. These are compared against with > and an operator who
  -- types 80 expects 80, not 79.99999999999999.
  `threshold`     DECIMAL(12,4)  NOT NULL,

  `is_active`     TINYINT(1)     NOT NULL DEFAULT 1,

  -- Read AND written by the evaluator, to stop a rule that stays true firing
  -- once a minute forever. The cooldown window lives in the job, not here, so
  -- it can be tuned without a migration.
  `last_fired_at` DATETIME       NULL DEFAULT NULL,

  `created_at`    DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (`id`),
  -- The evaluator's own sweep: every armed rule, nothing else.
  KEY `idx_alert_rules_active` (`is_active`, `metric`),
  KEY `idx_alert_rules_client` (`client_id`),
  KEY `idx_alert_rules_site` (`site_id`)

  -- No FKs to clients/sites. A rule outliving a deleted site is a dangling row
  -- the console can show and an operator can delete; CASCADE would silently
  -- remove alerting configuration as a side effect of tidying up a tenant, and
  -- nobody would notice until the alert that should have fired did not.
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
