/** Shapes returned by GET /api/admin/tenants/{id}.
 *
 * Kept here rather than in the screen so the action definitions can import them
 * without importing the screen back — a cycle that TypeScript tolerates and
 * bundlers punish in ways that are miserable to debug.
 */

export interface Site {
  id: string; domain: string; platform: string; platform_version: string | null;
  store_name: string | null; collection_name: string | null; environment: string;
  index_plan: string; catalogue_limit: number; indexed_items: number;
  catalogue_pct: number | null; is_active: boolean; created_at: string | null;
}

export interface Subscription {
  id: string; site_id: string; domain: string; environment: string;
  product_code: string; status: string; plan: string; request_limit: number;
  disabled_reason: string | null; expires_at: string | null; active_licences: number;
}

export interface Licence {
  id: string; key_prefix: string | null; has_plaintext: boolean; is_active: boolean;
  product_code: string; domain: string; environment: string;
  issued_at: string | null; expires_at: string | null; revoked_at: string | null;
}

export interface Client {
  id: string; name: string; email: string; company: string | null;
  plan: string | null; is_active: boolean; created_at: string | null;
}

export interface TenantDetailData {
  client: Client;
  sites: Site[];
  subscriptions: Subscription[];
  licences: Licence[];
  series: { day: string; requests: number; cost: number; tokens: number }[];
  by_product: { product_code: string; requests: number; cost: number; tokens: number }[];
  by_model: { provider: string; model: string; cost: number; tokens: number; calls: number }[];
  window_days: number;
}
