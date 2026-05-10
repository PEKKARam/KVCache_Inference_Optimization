wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

request_body = [[
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt": "Explain KV cache in one paragraph.",
  "max_tokens": 128,
  "temperature": 0.0
}
]]

function request()
  return wrk.format(nil, "/v1/completions", nil, request_body)
end
