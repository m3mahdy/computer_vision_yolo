"""
Delete weights, wandb, and val folders from tuning trials and generate a cleanup report.

This script recursively scans the tuning directory for 'weights', 'wandb', and 'val' folders,
calculates their sizes, deletes them, and generates a detailed report.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path


def get_folder_size(folder_path: Path) -> int:
    """Calculate total size of a folder in bytes."""
    total_size = 0
    try:
        for item in folder_path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not access some files in {folder_path}: {e}")
    return total_size


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def find_weights_folders(base_dir: Path) -> list[dict]:
    """Find all 'weights', 'wandb', and 'val' folders recursively."""
    target_folders = []
    
    # Find weights folders
    for weights_path in base_dir.rglob('weights'):
        if weights_path.is_dir():
            size_bytes = get_folder_size(weights_path)
            files = list(weights_path.rglob('*'))
            file_count = sum(1 for f in files if f.is_file())
            
            target_folders.append({
                'path': str(weights_path),
                'relative_path': str(weights_path.relative_to(base_dir)),
                'folder_type': 'weights',
                'size_bytes': size_bytes,
                'size_formatted': format_size(size_bytes),
                'file_count': file_count,
                'files': [str(f.relative_to(weights_path)) for f in files if f.is_file()]
            })
    
    # Find wandb folders
    for wandb_path in base_dir.rglob('wandb'):
        if wandb_path.is_dir():
            size_bytes = get_folder_size(wandb_path)
            files = list(wandb_path.rglob('*'))
            file_count = sum(1 for f in files if f.is_file())
            
            target_folders.append({
                'path': str(wandb_path),
                'relative_path': str(wandb_path.relative_to(base_dir)),
                'folder_type': 'wandb',
                'size_bytes': size_bytes,
                'size_formatted': format_size(size_bytes),
                'file_count': file_count,
                'files': [str(f.relative_to(wandb_path)) for f in files if f.is_file()]
            })
    
    # Find val folders
    for val_path in base_dir.rglob('val'):
        if val_path.is_dir():
            size_bytes = get_folder_size(val_path)
            files = list(val_path.rglob('*'))
            file_count = sum(1 for f in files if f.is_file())
            
            target_folders.append({
                'path': str(val_path),
                'relative_path': str(val_path.relative_to(base_dir)),
                'folder_type': 'val',
                'size_bytes': size_bytes,
                'size_formatted': format_size(size_bytes),
                'file_count': file_count,
                'files': [str(f.relative_to(val_path)) for f in files if f.is_file()]
            })
    
    return target_folders


def delete_weights_folders(weights_folders: list[dict], dry_run: bool = False) -> dict:
    """Delete weights, wandb, and val folders and return deletion report."""
    deleted = []
    failed = []
    total_size_freed = 0
    
    for folder_info in weights_folders:
        folder_path = Path(folder_info['path'])
        
        try:
            if not dry_run:
                shutil.rmtree(folder_path)
            
            deleted.append({
                'path': folder_info['relative_path'],
                'type': folder_info['folder_type'],
                'size': folder_info['size_formatted'],
                'size_bytes': folder_info['size_bytes'],
                'file_count': folder_info['file_count'],
                'files': folder_info['files']
            })
            total_size_freed += folder_info['size_bytes']
            
        except Exception as e:
            failed.append({
                'path': folder_info['relative_path'],
                'type': folder_info['folder_type'],
                'error': str(e)
            })
    
    return {
        'deleted': deleted,
        'failed': failed,
        'total_size_freed': format_size(total_size_freed),
        'total_size_freed_bytes': total_size_freed,
        'deleted_count': len(deleted),
        'failed_count': len(failed)
    }


def generate_report(report_data: dict, output_path: Path, dry_run: bool = False):
    """Generate and save cleanup report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Console report
    print('\n' + '=' * 80)
    print(f'WEIGHTS, WANDB & VAL CLEANUP REPORT {"(DRY RUN)" if dry_run else ""}')
    print('=' * 80)
    print(f'Timestamp: {timestamp}')
    print(f'Total folders {"found" if dry_run else "deleted"}: {report_data["deleted_count"]}')
    print(f'Failed deletions: {report_data["failed_count"]}')
    print(f'Total space {"to be freed" if dry_run else "freed"}: {report_data["total_size_freed"]}')
    print('=' * 80)
    
    if report_data['deleted']:
        print(f'\n✓ Successfully {"Found" if dry_run else "Deleted"} Folders:')
        print('-' * 80)
        for i, item in enumerate(report_data['deleted'], 1):
            print(f'\n{i}. [{item["type"].upper()}] {item["path"]}')
            print(f'   Size: {item["size"]} ({item["file_count"]} files)')
            if item['files']:
                print(f'   Files: {", ".join(item["files"][:5])}{"..." if len(item["files"]) > 5 else ""}')
    
    if report_data['failed']:
        print(f'\n✗ Failed Deletions:')
        print('-' * 80)
        for i, item in enumerate(report_data['failed'], 1):
            print(f'{i}. [{item["type"].upper()}] {item["path"]}')
            print(f'   Error: {item["error"]}')
    
    print('\n' + '=' * 80)
    
    # Save JSON report
    json_report = {
        'timestamp': timestamp,
        'dry_run': dry_run,
        'summary': {
            'deleted_count': report_data['deleted_count'],
            'failed_count': report_data['failed_count'],
            'total_size_freed': report_data['total_size_freed'],
            'total_size_freed_bytes': report_data['total_size_freed_bytes']
        },
        'deleted_folders': report_data['deleted'],
        'failed_deletions': report_data['failed']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2)
    
    print(f'\n📄 Report saved to: {output_path}')


def main():
    """Main execution function."""
    # Configuration
    BASE_DIR = Path(__file__).parent
    TUNE_DIR = BASE_DIR / 'tune'
    
    # Check if tune directory exists
    if not TUNE_DIR.exists():
        print(f'Error: Tune directory not found: {TUNE_DIR}')
        return
    
    print(f'\n🔍 Scanning for weights, wandb, and val folders in: {TUNE_DIR}')
    
    # Find all weights folders
    weights_folders = find_weights_folders(TUNE_DIR)
    
    if not weights_folders:
        print('\n✓ No weights, wandb, or val folders found!')
        return
    
    print(f'\n📊 Found {len(weights_folders)} folders (weights + wandb + val)')
    total_size = sum(f['size_bytes'] for f in weights_folders)
    print(f'📦 Total size: {format_size(total_size)}')
    
    # Ask for confirmation
    print('\n' + '=' * 80)
    print('⚠️  WARNING: This will permanently delete all weights, wandb, and val folders!')
    print('=' * 80)
    
    response = input('\nDo you want to proceed? (yes/no): ').strip().lower()
    
    if response not in ['yes', 'y']:
        print('\n❌ Operation cancelled by user')
        
        # Generate dry-run report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = BASE_DIR / f'weights_cleanup_dryrun_{timestamp}.json'
        
        report_data = {
            'deleted': [{
                'path': f['relative_path'],
                'type': f['folder_type'],
                'size': f['size_formatted'],
                'size_bytes': f['size_bytes'],
                'file_count': f['file_count'],
                'files': f['files']
            } for f in weights_folders],
            'failed': [],
            'total_size_freed': format_size(total_size),
            'total_size_freed_bytes': total_size,
            'deleted_count': len(weights_folders),
            'failed_count': 0
        }
        
        generate_report(report_data, report_path, dry_run=True)
        return
    
    # Perform deletion
    print('\n🗑️  Deleting weights, wandb, and val folders...')
    report_data = delete_weights_folders(weights_folders, dry_run=False)
    
    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = BASE_DIR / f'weights_cleanup_report_{timestamp}.json'
    generate_report(report_data, report_path, dry_run=False)
    
    print('\n✅ Cleanup completed successfully!')


if __name__ == '__main__':
    main()
