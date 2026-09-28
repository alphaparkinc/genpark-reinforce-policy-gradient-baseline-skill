import sys
import json
from client import REINFORCEPolicyGradient

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-reinforce-policy-gradient-baseline-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "train_reinforce_step",
                        "description": "Run REINFORCE policy gradient update over an episode trajectory",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "trajectory": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "description": "[state, action, reward]"
                                    }
                                },
                                "num_states": {"type": "integer", "default": 4},
                                "num_actions": {"type": "integer", "default": 2}
                            },
                            "required": ["trajectory"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "train_reinforce_step":
            traj = [tuple(step) for step in args.get("trajectory", [])]
            ns = args.get("num_states", 4)
            na = args.get("num_actions", 2)
            agent = REINFORCEPolicyGradient(num_states=ns, num_actions=na)
            agent.update_trajectory(traj)
            all_probs = [agent.get_action_probs(s) for s in range(ns)]
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"action_probabilities": all_probs})}]
                }
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
