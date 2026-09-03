-- 001 — licences: store the full plaintext key, drop the prefix column
--
-- Written 2026-09-03. Reverses the original v2 decision to store only a
-- SHA-256 hash plus a short display prefix. Rationale, and what it costs, is in
-- the comment block above LICENCES_TABLE in backend/app/services/schema_v2.py —
-- read it before running this. In short: this table becomes a secrets table.
-- Restrict SELECT on it, and keep dumps of it inside the building.
--
-- key_hash is UNTOUCHED and remains the only column resolution matches on.
--
-- IDEMPOTENT. Safe to run twice; safe to run against a database that already
-- has the column. MySQL 5.7/8.0 have no ADD COLUMN IF NOT EXISTS, hence the
-- information_schema guards.
--
-- NOT REVERSIBLE IN THE WAY THAT MATTERS. Dropping the column later removes the
-- plaintext from the live table but not from any backup taken while it existed.
--
-- THE SEVEN EXISTING LICENCES GET NULL. Their plaintext was never stored and
-- cannot be recovered — SHA-256 is one-way and there is nothing to backfill
-- from. Those keys keep working (resolution is by hash), but they can never be
-- displayed. The only way to give a store a readable key is to reissue, which
-- is already planned as ADMIN_CONSOLE_PLAN.md §4.1. Do not let anyone "fix"
-- the NULLs by inventing values: a wrong plaintext beside a correct hash is a
-- row that lies, and it will be believed.

-- ── 1. Add the plaintext column ──────────────────────────────────────────────

SET @col_exists := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME   = 'licences'
    AND COLUMN_NAME  = 'licence_key'
);

SET @sql := IF(@col_exists = 0,
  'ALTER TABLE `licences`
     ADD COLUMN `licence_key` VARCHAR(128)
       CHARACTER SET ascii COLLATE ascii_bin NULL
       COMMENT ''SECRET. Full plaintext key, for display and recovery only - never authorise on it. NULL for keys minted before 2026-09-03.''
       AFTER `key_hash`',
  'SELECT ''licences.licence_key already present, skipping'' AS note'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ── 2. Print what the drop is about to destroy ───────────────────────────────
--
-- For a licence minted after this migration, prefix_of(licence_key) reproduces
-- key_prefix exactly, so the column is redundant. For the seven minted BEFORE
-- it, licence_key is NULL and there is nothing to derive a prefix from — so
-- dropping the column takes away the only label those rows have, and the only
-- way to match a key already pasted into a merchant's config back to its row.
--
-- That is a real loss, it is small, and it ends the moment those licences are
-- reissued (ADMIN_CONSOLE_PLAN.md §4.1). Printing the mapping first means it
-- costs a scrollback search rather than being gone. Keep this output.

SELECT
  l.id            AS licence_id,
  l.key_prefix    AS label_being_dropped,
  s.product_code  AS product,
  si.domain       AS domain,
  l.is_active     AS active
FROM licences l
JOIN subscriptions s ON s.id = l.subscription_id
JOIN sites si        ON si.id = s.site_id
WHERE l.licence_key IS NULL
ORDER BY si.domain, s.product_code;

-- ── 3. Drop the prefix column ────────────────────────────────────────────────
--
-- Separate guard rather than one combined ALTER: on a re-run after a partial
-- failure the two halves may legitimately be in different states, and a single
-- statement would then either skip the outstanding half or error on the done
-- one. Everything that displayed this value now derives it from the plaintext
-- via license_key.prefix_of().

SET @prefix_exists := (
  SELECT COUNT(*) FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE()
    AND TABLE_NAME   = 'licences'
    AND COLUMN_NAME  = 'key_prefix'
);

SET @sql := IF(@prefix_exists = 1,
  'ALTER TABLE `licences` DROP COLUMN `key_prefix`',
  'SELECT ''licences.key_prefix already dropped, skipping'' AS note'
);
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ── 4. Report ────────────────────────────────────────────────────────────────

SELECT
  COUNT(*)                                        AS licences_total,
  SUM(licence_key IS NOT NULL)                    AS with_plaintext,
  SUM(licence_key IS NULL)                        AS needs_reissue_to_be_displayable
FROM licences;
