-- 004 — licence_events: what happened to this key, including with no human involved
--
-- ADMIN_CONSOLE_PLAN.md §5.3. Deliberately NOT the same table as
-- admin_audit_log, and the split is not tidiness:
--
--   admin_audit_log answers "WHO DID IT" and every row has an actor.
--   licence_events  answers "WHAT HAPPENED TO THIS KEY" and some rows have no
--                   actor at all — an expiry is nobody's action.
--
-- Folding them together would mean either inventing a fake actor for expiry
-- rows or making actor_email nullable in the table whose entire purpose is
-- attributing actions to people.
--
-- THIS TABLE IS ALSO THE ANSWER TO A HOLE WE OPENED. issue_licence() deletes
-- the licence it rotates out, so a rotated key now hashes to no row and
-- "rotated out" is indistinguishable from "never issued" — fine on dev stores,
-- not fine once a real customer rings up about a key that stopped working. An
-- event row survives the deletion of the licence it describes and is what makes
-- that question answerable again. Which is why licence_id is NOT a foreign key.

CREATE TABLE IF NOT EXISTS `licence_events` (
  `id`              BIGINT      NOT NULL AUTO_INCREMENT,

  -- NOT a foreign key, on purpose and load-bearing. A rotation DELETEs the
  -- licences row; ON DELETE CASCADE would take the history with it and ON
  -- DELETE RESTRICT would make rotation fail outright. The id is kept as a
  -- correlation handle that may point at a row which no longer exists.
  `licence_id`      VARCHAR(36) NOT NULL,

  -- Survives the licence. A subscription is the thing an operator is actually
  -- looking at when they ask what happened, and unlike licence_id it is still
  -- resolvable after a rotation.
  `subscription_id` VARCHAR(36) NOT NULL,

  -- The prefix, never the key. licences.licence_key holds the plaintext now and
  -- this table has no business duplicating it into a second place with
  -- different retention — this one is never purged.
  `key_prefix`      VARCHAR(32) NULL DEFAULT NULL
                    COMMENT 'czg_test_mpqa_7Kq2, from license_key.prefix_of(). Identifies the key in a timeline; useless for authenticating.',

  `event`           VARCHAR(32) NOT NULL
                    COMMENT 'issued | rotated | superseded | revoked | expired',
  `detail`          VARCHAR(500) NULL DEFAULT NULL,

  -- Who, when there was a who. Same denormalisation as admin_audit_log and for
  -- the same reason; NULL for events with no human actor, which is the case
  -- this table exists to hold.
  `actor_email`     VARCHAR(255) NULL DEFAULT NULL,

  `created_at`      DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (`id`),
  KEY `idx_licence_events_licence` (`licence_id`, `created_at`),
  -- The index that still works after a rotation has deleted the licence row.
  KEY `idx_licence_events_subscription` (`subscription_id`, `created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
