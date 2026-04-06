import sys

from av_nav.config import get_config

try:
    from habitat.datasets import make_dataset
except ImportError:
    make_dataset = None

CFG_PATH = "av_nav/config/mp3d/train_telephone/audiogoal_depth.yaml"

def main():
    cfg = get_config(CFG_PATH)
    print("=== CONFIG CHECK ===")
    print("DATASET.TYPE:", cfg.DATASET.TYPE)
    print("DATASET.DATA_PATH:", cfg.DATASET.DATA_PATH)
    print("DATASET.CONTENT_SCENES:", cfg.DATASET.CONTENT_SCENES)

    scenes = None
    episodes = None

    if make_dataset is not None:
        try:
            dataset = make_dataset(cfg.DATASET.TYPE, config=cfg.DATASET)
            episodes = dataset.episodes
            scenes = sorted({ep.scene_id for ep in episodes})
        except Exception as e:
            print("[WARN] habitat.make_dataset failed:", repr(e))

    if scenes is None:
        cs = cfg.DATASET.CONTENT_SCENES
        if isinstance(cs, (list, tuple)) and cs and cs != ["*"]:
            scenes = cs
        else:
            scenes = []

    print("\n=== DATASET SUMMARY ===")
    print(f"Scenes detected: {len(scenes)}")
    for s in scenes:
        print(" -", s)
    if episodes is not None:
        print(f"Episodes detected: {len(episodes)}")
    print("\nDone.")

if __name__ == "__main__":
    main()
