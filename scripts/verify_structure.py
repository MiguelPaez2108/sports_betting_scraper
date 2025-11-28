"""
Project structure verification script.
Checks that all necessary files and directories exist.
"""
import os
from pathlib import Path
from typing import List, Tuple


def check_directories(base_path: Path, dirs: List[str]) -> Tuple[int, int]:
    """Check if directories exist"""
    exists = 0
    missing = 0
    
    for dir_path in dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            exists += 1
        else:
            print(f"[MISSING] Directory: {dir_path}")
            missing += 1
    
    return exists, missing


def check_files(base_path: Path, files: List[str]) -> Tuple[int, int]:
    """Check if files exist"""
    exists = 0
    missing = 0
    
    for file_path in files:
        full_path = base_path / file_path
        if full_path.exists():
            exists += 1
        else:
            print(f"[MISSING] File: {file_path}")
            missing += 1
    
    return exists, missing


def main():
    base_path = Path(__file__).parent.parent
    
    print("=" * 60)
    print("Verifying Sports Betting Intelligence Platform Structure")
    print("=" * 60)
    print()
    
    # Critical directories
    critical_dirs = [
        "src/prediction",
        "src/data_collection",
        "src/feature_engineering",
        "src/recommendation",
        "src/backtesting",
        "src/api",
        "src/shared",
        "frontend/src",
        "tests",
        "infrastructure/docker",
        "models",
        "data"
    ]
    
    # Critical files
    critical_files = [
        "pyproject.toml",
        "docker-compose.yml",
        ".env",
        ".env.example",
        ".gitignore",
        "README.md",
        "Makefile",
        "src/shared/config/settings.py",
        "src/shared/logging/logger.py",
        "src/shared/database/postgres.py",
        "src/shared/database/mongodb.py",
        "src/shared/database/redis.py",
        "infrastructure/docker/Dockerfile.backend",
        "infrastructure/docker/Dockerfile.celery",
        "infrastructure/nginx/nginx.conf"
    ]
    
    # Check directories
    print("[1/2] Checking directories...")
    dir_exists, dir_missing = check_directories(base_path, critical_dirs)
    print(f"[OK] {dir_exists}/{len(critical_dirs)} directories exist\n")
    
    # Check files
    print("[2/2] Checking files...")
    file_exists, file_missing = check_files(base_path, critical_files)
    print(f"[OK] {file_exists}/{len(critical_files)} files exist\n")
    
    # Summary
    total_items = len(critical_dirs) + len(critical_files)
    total_exists = dir_exists + file_exists
    total_missing = dir_missing + file_missing
    
    print("=" * 60)
    print("Summary:")
    print(f"   Total items checked: {total_items}")
    print(f"   [OK] Exists: {total_exists}")
    print(f"   [MISSING] Missing: {total_missing}")
    print(f"   Completion: {(total_exists/total_items)*100:.1f}%")
    print("=" * 60)
    
    if total_missing == 0:
        print("\n[SUCCESS] All critical files and directories are in place!")
        print("[SUCCESS] Project structure is ready for development!")
    else:
        print(f"\n[WARNING] {total_missing} items are missing. Please review above.")
    
    # Configuration check
    print("\nConfiguration Status:")
    env_file = base_path / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if "FOOTBALL_DATA_API_KEY=3e45363c9c074c9d9254a155d6f89020" in content:
                print("   [OK] Football-Data.org API key configured")
            if "ODDS_API_KEY=aa55ec5be4748f07a76766ac6f1efc19" in content:
                print("   [OK] The Odds API key configured")
            if "LEAGUE_LALIGA_ID=2014" in content:
                print("   [OK] La Liga configured")
            if "LEAGUE_SERIEA_ID=2019" in content:
                print("   [OK] Serie A configured")
            if "LEAGUE_PREMIER_ID=2021" in content:
                print("   [OK] Premier League configured")
            if "LEAGUE_BUNDESLIGA_ID=2002" in content:
                print("   [OK] Bundesliga configured")
            if "LEAGUE_CHAMPIONS_ID=2001" in content:
                print("   [OK] Champions League configured")
    
    print("\nNext Steps:")
    print("   1. Install dependencies: poetry install")
    print("   2. Start Docker services: docker-compose up -d")
    print("   3. Run tests: poetry run pytest")
    print("   4. Start development: make run")


if __name__ == "__main__":
    main()
