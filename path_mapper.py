"""
Utility to remap file paths from Kaggle to local VM paths
Handles path transformation when running locally instead of on Kaggle
"""

import os
from pathlib import Path


class PathMapper:
    """Maps Kaggle paths to local VM paths"""
    
    def __init__(self, kaggle_prefix="/kaggle/input", local_base_path=None):
        """
        Args:
            kaggle_prefix: The Kaggle path prefix to replace (default: /kaggle/input)
            local_base_path: Your local base path (e.g., /media/vhviet03/datasets)
        """
        self.kaggle_prefix = kaggle_prefix
        self.local_base_path = local_base_path
    
    def extract_filename(self, kaggle_path):
        """Extract just the filename from a full Kaggle path
        
        Args:
            kaggle_path: Full path like "/kaggle/input/rsna-png/data/train/SARS-10.jpeg"
        
        Returns:
            Filename like "SARS-10.jpeg"
        """
        return os.path.basename(kaggle_path)
    
    def extract_relative_path(self, kaggle_path):
        """Extract relative path after the dataset name
        
        Args:
            kaggle_path: "/kaggle/input/rsna-png/data/train/SARS-10.jpeg"
        
        Returns:
            "data/train/SARS-10.jpeg"
        """
        # Split path and find the relative part after /kaggle/input/{dataset_name}/
        parts = kaggle_path.split('/')
        
        # Find index of 'input' in path
        if 'input' in parts:
            input_idx = parts.index('input')
            # Take everything after /kaggle/input/{dataset_name}/
            # dataset_name is at input_idx + 1
            relative_parts = parts[input_idx + 2:]  # Skip 'input' and 'dataset_name'
            return '/'.join(relative_parts)
        
        # Fallback: just return filename
        return self.extract_filename(kaggle_path)
    
    def remap_path(self, kaggle_path, local_base_path=None):
        """Remap Kaggle path to local VM path
        
        Examples:
            kaggle_path: "/kaggle/input/rsna-png/data/train/SARS-10.jpeg"
            local_base_path: "/media/vhviet03/datasets/rsna-png"
            returns: "/media/vhviet03/datasets/rsna-png/data/train/SARS-10.jpeg"
            
            OR with filename only:
            kaggle_path: "/kaggle/input/rsna-png/data/train/SARS-10.jpeg"
            local_base_path: "/media/vhviet03/datasets/covidx-cxr/data/train"
            returns: "/media/vhviet03/datasets/covidx-cxr/data/train/SARS-10.jpeg"
        """
        base_path = local_base_path or self.local_base_path
        
        if not base_path:
            raise ValueError("local_base_path must be provided")
        
        filename = self.extract_filename(kaggle_path)
        remapped = os.path.join(base_path, filename)
        
        return remapped
    
    def verify_path(self, kaggle_path, local_base_path=None):
        """Check if remapped path exists and return it
        
        Args:
            kaggle_path: Original Kaggle path
            local_base_path: Local base path
        
        Returns:
            (exists: bool, remapped_path: str)
        """
        remapped = self.remap_path(kaggle_path, local_base_path)
        exists = os.path.exists(remapped)
        
        return exists, remapped
    
    def batch_remap(self, kaggle_paths, local_base_path=None):
        """Remap multiple paths
        
        Args:
            kaggle_paths: List of Kaggle paths
            local_base_path: Local base path
        
        Returns:
            List of remapped paths
        """
        return [self.remap_path(p, local_base_path) for p in kaggle_paths]


# Example usage
if __name__ == '__main__':
    # Initialize mapper
    mapper = PathMapper(
        kaggle_prefix="/kaggle/input",
        local_base_path="/media/vhviet03/datasets/covidx-cxr/data/train"
    )
    
    # Test path
    kaggle_path = "/kaggle/input/rsna-png/data/train/SARS-10.1148rg.242035193-g04mr34g0-Fig8a-day0.jpeg"
    
    print("=" * 70)
    print("PATH MAPPING TEST")
    print("=" * 70)
    print(f"\nOriginal Kaggle path:")
    print(f"  {kaggle_path}")
    
    # Extract filename
    filename = mapper.extract_filename(kaggle_path)
    print(f"\nExtracted filename:")
    print(f"  {filename}")
    
    # Remap path
    remapped = mapper.remap_path(kaggle_path)
    print(f"\nRemapped local path:")
    print(f"  {remapped}")
    
    # Verify
    exists, verified_path = mapper.verify_path(kaggle_path)
    print(f"\nVerification:")
    print(f"  File exists: {exists}")
    print(f"  Path: {verified_path}")
    
    if not exists:
        print(f"\n⚠️  File not found at: {verified_path}")
        print(f"   Try these alternatives:")
        print(f"   1. Check if base path is correct:")
        print(f"      ls -la /media/vhviet03/datasets/covidx-cxr/data/train/ | head")
        print(f"   2. Use different base path for different datasets")
