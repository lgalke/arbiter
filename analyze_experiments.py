#!/usr/bin/env python3
"""
Analyze experiment statistics:
- Tool usage by setup and conversation
- Budget usage
- Wait and observe steps before final assessment
- Saves results to JSON
"""
import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

def analyze_experiments():
    """Extract and analyze experiment statistics."""
    results = {}
    
    experiments_dir = Path("results/v0.5")
    
    # Group by cell (conversation_setup_budget combination)
    cells = defaultdict(lambda: {
        "conversation": "",
        "tool_setup": "",
        "budget": 0,
        "runs": [],
        "stats": {
            "total_runs": 0,
            "total_budget_used": 0,
            "avg_budget_used": 0.0,
            "max_budget_used": 0,
            "min_budget_used": float('inf'),
            "tool_usage": defaultdict(int),
            "tool_usage_pct": {},
            "wait_and_observe_counts": [],
            "avg_wait_and_observe": 0.0,
            "max_wait_and_observe": 0,
            "min_wait_and_observe": float('inf'),
        }
    })
    
    # Process each cell directory
    for cell_dir in sorted(experiments_dir.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name == "exp_summary.json":
            continue
        
        cell_name = cell_dir.name
        
        # Find all result files for this cell
        result_files = sorted(cell_dir.glob("r*.json"))
        
        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            
            # Extract metadata
            conversation = data.get("input_file", "")
            budget = data.get("budget", 0)
            budget_used = data.get("budget_used", 0)
            interactions = data.get("interactions", [])
            
            # Count tool usage
            tool_counts = defaultdict(int)
            wait_and_observe_count = 0
            
            for interaction in interactions:
                tool = interaction.get("tool", "unknown")
                tool_counts[tool] += 1
                if tool == "wait_and_observe":
                    wait_and_observe_count += 1
            
            # Store run data
            cells[cell_name]["conversation"] = conversation
            cells[cell_name]["budget"] = budget
            cells[cell_name]["runs"].append({
                "rep": result_file.stem,
                "budget_used": budget_used,
                "tool_counts": dict(tool_counts),
                "wait_and_observe_count": wait_and_observe_count,
            })
    
    # Calculate statistics per cell
    for cell_name, cell_data in cells.items():
        stats = cell_data["stats"]
        runs = cell_data["runs"]
        
        if not runs:
            continue
        
        # Total and average budget
        stats["total_runs"] = len(runs)
        budget_used_list = [r["budget_used"] for r in runs]
        stats["total_budget_used"] = sum(budget_used_list)
        stats["avg_budget_used"] = stats["total_budget_used"] / len(runs) if runs else 0
        stats["max_budget_used"] = max(budget_used_list) if budget_used_list else 0
        stats["min_budget_used"] = min(budget_used_list) if budget_used_list else 0
        
        # Tool usage aggregation per tool (for SD calculation)
        tool_usage_per_run = defaultdict(list)
        all_tool_counts = defaultdict(int)
        
        for run in runs:
            for tool, count in run["tool_counts"].items():
                all_tool_counts[tool] += count
                tool_usage_per_run[tool].append(count)
        
        stats["tool_usage"] = dict(all_tool_counts)
        stats["tool_usage_per_run"] = dict(tool_usage_per_run)
        
        # Calculate percentages
        total_tools = sum(all_tool_counts.values())
        for tool, count in all_tool_counts.items():
            pct = (count / total_tools * 100) if total_tools > 0 else 0
            stats["tool_usage_pct"][tool] = round(pct, 1)
        
        # Wait and observe statistics
        wait_counts = [r["wait_and_observe_count"] for r in runs]
        stats["wait_and_observe_counts"] = wait_counts
        stats["avg_wait_and_observe"] = sum(wait_counts) / len(wait_counts) if wait_counts else 0
        stats["sd_wait_and_observe"] = statistics.stdev(wait_counts) if len(wait_counts) > 1 else 0
        stats["max_wait_and_observe"] = max(wait_counts) if wait_counts else 0
        stats["min_wait_and_observe"] = min(wait_counts) if wait_counts else 0
        
        # Remove non-serializable defaultdicts
        stats["tool_usage"] = dict(stats["tool_usage"])
        stats["tool_usage_pct"] = dict(stats["tool_usage_pct"])
        stats["tool_usage_per_run"] = dict(stats["tool_usage_per_run"])
        del stats["tool_usage_pct"]  # Remove percentages, recalculate on output
    
    # Convert to serializable format
    output = {}
    for cell_name, cell_data in sorted(cells.items()):
        if cell_data["runs"]:
            stats = cell_data["stats"]
            
            # Calculate per-tool statistics (mean and SD)
            tool_stats = {}
            for tool, usage_list in stats["tool_usage_per_run"].items():
                mean = sum(usage_list) / len(usage_list) if usage_list else 0
                sd = statistics.stdev(usage_list) if len(usage_list) > 1 else 0
                tool_stats[tool] = {
                    "total": stats["tool_usage"].get(tool, 0),
                    "mean": round(mean, 2),
                    "sd": round(sd, 2),
                }
            
            output[cell_name] = {
                "conversation": cell_data["conversation"],
                "budget": cell_data["budget"],
                "total_runs": stats["total_runs"],
                "budget_stats": {
                    "total_used": stats["total_budget_used"],
                    "average": round(stats["avg_budget_used"], 2),
                    "max": stats["max_budget_used"],
                    "min": stats["min_budget_used"],
                },
                "tool_usage": {
                    "per_tool_stats": tool_stats,
                    "total_calls": stats["tool_usage"],
                    "percentages": {
                        tool: round(count / sum(stats["tool_usage"].values()) * 100, 1)
                        if sum(stats["tool_usage"].values()) > 0 else 0
                        for tool, count in stats["tool_usage"].items()
                    },
                },
                "wait_and_observe": {
                    "mean": round(stats["avg_wait_and_observe"], 2),
                    "sd": round(stats["sd_wait_and_observe"], 2),
                    "max": stats["max_wait_and_observe"],
                    "min": stats["min_wait_and_observe"],
                    "per_run": stats["wait_and_observe_counts"],
                },
            }
    
    # Save to file
    output_file = Path("results/v0.5/analysis_stats.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Statistics saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT STATISTICS SUMMARY")
    print("="*80)
    
    for cell_name, stats in sorted(output.items()):
        print(f"\n{cell_name}:")
        print(f"  Conversation: {stats['conversation']}")
        print(f"  Total Runs: {stats['total_runs']}")
        print(f"  Budget (per run): {stats['budget']}")
        print(f"  Budget Stats:")
        print(f"    - Average used: {stats['budget_stats']['average']}")
        print(f"    - Range: {stats['budget_stats']['min']} - {stats['budget_stats']['max']}")
        print(f"  Tool Usage (per-run mean ± SD):")
        for tool, tool_stat in sorted(stats['tool_usage']['per_tool_stats'].items(), key=lambda x: -x[1]['total']):
            total = tool_stat['total']
            mean = tool_stat['mean']
            sd = tool_stat['sd']
            pct = stats['tool_usage']['percentages'].get(tool, 0)
            print(f"    - {tool}: {total} total ({mean} ± {sd} per run, {pct}%)")
        print(f"  Wait and Observe Steps:")
        print(f"    - Mean ± SD: {stats['wait_and_observe']['mean']} ± {stats['wait_and_observe']['sd']}")
        print(f"    - Range: {stats['wait_and_observe']['min']} - {stats['wait_and_observe']['max']}")

if __name__ == "__main__":
    analyze_experiments()
