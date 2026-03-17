from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    # 获取项目根目录
    project_root = Path(__file__).resolve().parents[1]
    # 获取当前文件的目录
    # current_dir = Path(__file__).resolve().parent
    print(f"app.py 获取的项目根目录是：{project_root}")

    # 根据项目根目录获取配置文件
    load_dotenv(
        dotenv_path = project_root / ".env",
        override = True,
        verbose = True,
        encoding = "utf-8",
    )
