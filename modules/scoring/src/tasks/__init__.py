from src.registry import register_all, rel_dir, get_all_tasks


register_all(rel_dir(__file__))
__all__ = ["get_all_tasks"]
