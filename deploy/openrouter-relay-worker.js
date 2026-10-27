// Cloudflare Worker that relays /api/v1/* to OpenRouter from a non-CN egress.
// Needed because openai/gpt-4o-mini via OpenRouter returns 403 "not available in your region"
// from the CN server. Deploy in the Cloudflare dashboard (Workers -> Create -> paste), set a
// secret RELAY_TOKEN, then point llm.base_url at https://<worker>.workers.dev/api/v1 and send
// the relay token as the llm.api_key is NOT used by OpenRouter here: put the real OpenRouter key
// in the Worker secret OPENROUTER_API_KEY and use RELAY_TOKEN as llm.api_key in config.local.yaml.
export default {
  async fetch(request, env) {
    const auth = request.headers.get("Authorization") || "";
    if (auth !== `Bearer ${env.RELAY_TOKEN}`) return new Response("unauthorized", { status: 401 });
    const url = new URL(request.url);
    if (!url.pathname.startsWith("/api/v1/")) return new Response("not found", { status: 404 });
    const headers = new Headers(request.headers);
    headers.set("Authorization", `Bearer ${env.OPENROUTER_API_KEY}`);
    headers.delete("host");
    return fetch("https://openrouter.ai" + url.pathname + url.search, {
      method: request.method,
      headers,
      body: request.body,
    });
  },
};
