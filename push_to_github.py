#!/usr/bin/env python3
"""Push local files to GitHub repo via Git Data API using gh cli."""
import subprocess
import json
import os
import sys
import base64
import tempfile

REPO = "haimianxing/synthesis-collapse"
BRANCH = "master"
WORKDIR = "/tmp/synthesis-collapse-update"

def create_blob(filepath):
    """Create a blob on GitHub and return SHA."""
    with open(filepath, 'rb') as f:
        content = f.read()
    
    is_binary = filepath.endswith(('.pdf', '.png', '.jpg'))
    
    data = {}
    if is_binary:
        data["content"] = base64.b64encode(content).decode()
        data["encoding"] = "base64"
    else:
        try:
            text = content.decode('utf-8')
            data["content"] = text
            data["encoding"] = "utf-8"
        except UnicodeDecodeError:
            data["content"] = base64.b64encode(content).decode()
            data["encoding"] = "base64"
    
    # Write to temp file to avoid stdin encoding issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump(data, tf)
        tf_path = tf.name
    
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/git/blobs", "--method", "POST", "--input", tf_path],
        capture_output=True, text=True, timeout=60
    )
    os.unlink(tf_path)
    
    if result.returncode != 0:
        print(f"  Blob error for {filepath}: {result.stderr[:200]}", file=sys.stderr)
        return None
    
    return json.loads(result.stdout)["sha"]

def api_post(endpoint, data):
    """POST to GitHub API via temp file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump(data, tf)
        tf_path = tf.name
    
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/{endpoint}", "--method", "POST", "--input", tf_path],
        capture_output=True, text=True, timeout=60
    )
    os.unlink(tf_path)
    
    if result.returncode != 0:
        print(f"  API error: {result.stderr[:300]}", file=sys.stderr)
        return None
    return json.loads(result.stdout)

def api_patch(endpoint, data):
    """PATCH to GitHub API via temp file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
        json.dump(data, tf)
        tf_path = tf.name
    
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/{endpoint}", "--method", "PATCH", "--input", tf_path],
        capture_output=True, text=True, timeout=60
    )
    os.unlink(tf_path)
    
    if result.returncode != 0:
        print(f"  API error: {result.stderr[:300]}", file=sys.stderr)
        return None
    return json.loads(result.stdout)

def api_get(endpoint):
    """GET from GitHub API."""
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/{endpoint}"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)

def collect_files(workdir, prefix=""):
    """Collect all files recursively."""
    files = []
    for item in sorted(os.listdir(workdir)):
        fullpath = os.path.join(workdir, item)
        relpath = f"{prefix}{item}" if prefix == "" else f"{prefix}/{item}"
        
        if os.path.isdir(fullpath):
            files.extend(collect_files(fullpath, relpath))
        elif os.path.isfile(fullpath):
            size = os.path.getsize(fullpath)
            if size < 10 * 1024 * 1024:
                files.append((relpath, fullpath, size))
    return files

def main():
    print("=== GitHub Repo Updater ===")
    
    # 1. Collect all files
    print("\n[1/5] Collecting files...")
    files = collect_files(WORKDIR)
    total_size = sum(s for _, _, s in files)
    print(f"  Found {len(files)} files ({total_size/1024/1024:.1f} MB)")
    
    # 2. Create blobs (batch to avoid rate limits)
    print("\n[2/5] Creating blobs...")
    tree_entries = []
    failed = 0
    for i, (relpath, fullpath, size) in enumerate(files):
        blob_sha = create_blob(fullpath)
        if blob_sha:
            tree_entries.append({
                "path": relpath,
                "mode": "100644",
                "type": "blob",
                "sha": blob_sha
            })
        else:
            failed += 1
        if (i + 1) % 30 == 0:
            print(f"  Progress: {i+1}/{len(files)} blobs ({len(tree_entries)} ok, {failed} failed)")
    
    print(f"  Created {len(tree_entries)} blobs ({failed} failed)")
    
    if not tree_entries:
        print("ERROR: No blobs created, aborting.")
        return
    
    # 3. Create tree
    print("\n[3/5] Creating tree...")
    tree_result = api_post("git/trees", {"tree": tree_entries})
    if not tree_result:
        print("ERROR: Failed to create tree")
        return
    tree_sha = tree_result["sha"]
    print(f"  Tree SHA: {tree_sha}")
    
    # 4. Create commit
    print("\n[4/5] Creating commit...")
    ref_result = api_get(f"git/refs/heads/{BRANCH}")
    if not ref_result:
        print("ERROR: Failed to get current ref")
        return
    parent_sha = ref_result["object"]["sha"]
    
    commit_data = {
        "message": "Update: new title, architecture diagram, all scripts/results/figures\n\n"
                   "- Title: Synthesis Collapse: How Greedy Selection Destroys Behavioral Diversity in LLM Data Synthesis\n"
                   "- Added architecture diagram (fig_overview_neurips_v2)\n"
                   "- 93 experiment scripts (Code/Math/Dialogue)\n"
                   "- Key results: LoRA rank sweep, 8-seed downstream, noise injection\n"
                   "- 38 figures (PDF+PNG)\n"
                   "- Comprehensive README with reproduction instructions",
        "tree": tree_sha,
        "parents": [parent_sha]
    }
    commit_result = api_post("git/commits", commit_data)
    if not commit_result:
        print("ERROR: Failed to create commit")
        return
    commit_sha = commit_result["sha"]
    print(f"  Commit SHA: {commit_sha}")
    
    # 5. Update ref
    print("\n[5/5] Updating master ref...")
    ref_data = {"sha": commit_sha, "force": False}
    ref_result = api_patch(f"git/refs/heads/{BRANCH}", ref_data)
    
    if ref_result:
        print(f"\n✓ Successfully pushed to {REPO}:{BRANCH}")
        print(f"  Commit: {commit_sha}")
        print(f"  Files: {len(tree_entries)}")
        print(f"  URL: https://github.com/{REPO}")
    else:
        print(f"\n✗ Push failed")

if __name__ == "__main__":
    main()
