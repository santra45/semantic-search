-- 005 — subscriptions.disabled_reason: the sentence a merchant actually reads
--
-- ADMIN_CONSOLE_PLAN.md §5.4. When a module is paused, the storefront widget
-- hides itself and the 403 carries a merchant_message. That message has had
-- nowhere to live: the operator types a `reason` into the audit log, which is
-- internal and never leaves the console, and the merchant gets a generic
-- string. This column is the merchant-facing half of the same action.
--
-- TWO DIFFERENT FIELDS, DO NOT MERGE THEM. admin_audit_log.reason is written
-- for colleagues six months later ("non-payment, ticket 4412, chased twice").
-- This one is written for the customer and is rendered into their storefront.
-- One field serving both ends up either uselessly vague internally or leaking
-- an internal note onto a shop.
--
-- Guarded rather than plain ALTER: MySQL has no ADD COLUMN IF NOT EXISTS, and a
-- migration that dies on re-run is a migration that cannot be resumed after a
-- partial failure.

SET @col_exists := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME   = 'subscriptions'
    AND COLUMN_NAME  = 'disabled_reason'
);

SET @sql := IF(@col_exists = 0,
  'ALTER TABLE `subscriptions`
     ADD COLUMN `disabled_reason` VARCHAR(500) NULL DEFAULT NULL
       COMMENT ''MERCHANT-FACING. Rendered into the storefront and the 403 merchant_message. Not for internal notes - that is admin_audit_log.reason.''
       AFTER `status`',
  'SELECT ''subscriptions.disabled_reason already present, skipping'' AS note'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;
