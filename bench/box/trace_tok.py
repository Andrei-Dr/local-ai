"""Patch llama-moe-trace to also write the traced token ids to <MOE_TRACE_OUT>.tok (int32, in trace order).
Run on the box from the llama.cpp worktree root: python3 /ai/bench/trace_tok.py"""
import pathlib
p = pathlib.Path("examples/eval-callback/moe-trace.cpp"); s = p.read_text()
if ".tok" in s:
    print("already patched"); raise SystemExit
old = """        tokens.resize(max_trace_tokens);
    }
"""
new = """        tokens.resize(max_trace_tokens);
    }
    {
        // sidecar with the token ids, in trace order: lets offline tools key expert routing on token n-grams
        const std::string tok_path = std::string(out_path) + ".tok";
        if (FILE * tf = fopen(tok_path.c_str(), "wb")) {
            fwrite(tokens.data(), sizeof(llama_token), tokens.size(), tf);
            fclose(tf);
        }
    }
"""
assert s.count(old) == 1
p.write_text(s.replace(old, new))
print("patched")
