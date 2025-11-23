#!/usr/bin/env python3
"""
Git commit and push script.
Adds all changes, commits with a message, and pushes to remote.
"""

import subprocess
import sys
from datetime import datetime


def run_command(command, check=True):
    """
    Run a shell command and return the result.
    
    Args:
        command: Command string or list of command parts
        check: Whether to raise exception on error
    
    Returns:
        CompletedProcess object
    """
    if isinstance(command, str):
        command = command.split()
    
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False
    )
    
    if check and result.returncode != 0:
        print(f'❌ Command failed: {" ".join(command)}')
        print(f'Error: {result.stderr}')
        sys.exit(1)
    
    return result


def get_git_status():
    """Get current git status."""
    result = run_command('git status --short')
    return result.stdout.strip()


def get_current_branch():
    """Get current git branch name."""
    result = run_command('git branch --show-current')
    return result.stdout.strip()


def configure_git():
    """Configure git user identity and credentials."""
    print('\n📝 Configuring Git...')
    
    # Set user name
    run_command(['git', 'config', '--global', 'user.name', 'Mohammed Mahdy'], check=False)
    print('✓ User name set: Mohammed Mahdy')
    
    # Set user email
    run_command(['git', 'config', '--global', 'user.email', 'm3edris@gmail.com'], check=False)
    print('✓ User email set: m3edris@gmail.com')
    
    # Set credential helper (cache for 1 hour)
    run_command(['git', 'config', '--global', 'credential.helper', 'cache --timeout=3600'], check=False)
    print('✓ Credential helper configured')
    
    print()


def main():
    print('=' * 80)
    print('GIT COMMIT AND PUSH')
    print('=' * 80)
    
    # Check if we're in a git repository
    result = run_command('git rev-parse --git-dir', check=False)
    if result.returncode != 0:
        print('❌ Not a git repository')
        sys.exit(1)
    
    # Configure git identity
    configure_git()
    
    # Get current branch
    branch = get_current_branch()
    print(f'Current branch: {branch}')
    
    # Check for changes
    status = get_git_status()
    if not status:
        print('✓ No changes to commit')
        sys.exit(0)
    
    print(f'\nChanges to commit:')
    print(status)
    print()
    
    # Get commit message
    default_message = f'Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    commit_message = input(f'Enter commit message (default: "{default_message}"): ').strip()
    
    if not commit_message:
        commit_message = default_message
    
    print(f'\nCommit message: {commit_message}')
    print('=' * 80)
    
    # Add all changes
    print('Adding all changes...')
    run_command('git add -A')
    print('✓ Changes added')
    
    # Commit
    print('Committing...')
    run_command(['git', 'commit', '-m', commit_message])
    print('✓ Changes committed')
    
    # Push
    print(f'Pushing to remote ({branch})...')
    result = run_command(f'git push origin {branch}', check=False)
    
    if result.returncode != 0:
        print(f'⚠️  Push failed: {result.stderr}')
        print('\nTrying to set upstream...')
        result = run_command(f'git push -u origin {branch}', check=False)
        
        if result.returncode != 0:
            print(f'❌ Push failed: {result.stderr}')
            sys.exit(1)
    
    print('✓ Changes pushed successfully')
    print('=' * 80)
    print('✓ Done!')


if __name__ == '__main__':
    main()
