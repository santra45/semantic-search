-- 007 — chat_quality_daily: aggregates pushed from the merchant's own database
--
-- ADMIN_CONSOLE_PLAN.md §8.4. Conversations, messages and ratings live in the
-- MERCHANT's database and the privacy refactor put them there deliberately —
-- this backend's own chat_conversations / chat_messages / chat_feedback have no
-- writer at all any more. So the console cannot query chat quality; the plugins
-- push a daily rollup and it lands here.
--
-- COUNTS AND TIMINGS ONLY. No message text, no customer identifiers, no
-- queries. If a column is ever proposed here that holds something a shopper
-- typed, that is the privacy decision being reopened, not a schema tweak.
--
-- Nothing writes this yet — the endpoint and the five plugin releases are
-- Phase 5/6. The table lands now so the endpoint has somewhere to land the day
-- it ships, rather than needing a migration against a live database at the same
-- time as a plugin rollout.

CREATE TABLE IF NOT EXISTS `chat_quality_daily` (
  -- Keyed on the subscription, NOT on (client_id, product_code). The licence
  -- the plugin authenticates with resolves to exactly one subscription, so
  -- there is nothing to infer — and it stays correct when one client runs two
  -- stores on the same product, which the client+product key would collide.
  `subscription_id`   VARCHAR(36) NOT NULL,

  -- A Magento install can serve several store views from one licence, and they
  -- are worth telling apart. Defaulted rather than nullable: NULL in a primary
  -- key is a different row every time in some engines and a headache in all of
  -- them.
  `store_code`        VARCHAR(64) NOT NULL DEFAULT 'default',

  `date`              DATE        NOT NULL,

  `sessions`          INT         NOT NULL DEFAULT 0,
  `messages`          INT         NOT NULL DEFAULT 0,
  `avg_response_ms`   INT         NOT NULL DEFAULT 0,
  `p95_response_ms`   INT         NOT NULL DEFAULT 0,
  `rating_up`         INT         NOT NULL DEFAULT 0,
  `rating_down`       INT         NOT NULL DEFAULT 0,
  `escalations`       INT         NOT NULL DEFAULT 0,
  `zero_result_turns` INT         NOT NULL DEFAULT 0,

  `by_agent_json`     JSON        NULL DEFAULT NULL
                      COMMENT 'Turn counts per agent name. Names, not content.',

  `received_at`       DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,

  -- The composite PK is what makes the endpoint an UPSERT: a re-post for the
  -- same day corrects the row rather than duplicating it, so the plugin cron is
  -- safely re-runnable and the 30-day first-run backfill cannot double-count.
  PRIMARY KEY (`subscription_id`, `store_code`, `date`),

  -- "Chat quality across all tenants for last month" — the console's rollup,
  -- which does not filter by subscription and would otherwise scan.
  KEY `idx_chat_quality_date` (`date`)

  -- No FK to subscriptions: telemetry for a subscription later deleted is still
  -- true, and a rollup that vanishes because a tenant was tidied up takes the
  -- historical picture with it.
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
