import pickle
import logging
from pathlib import Path
from typing import Any, Tuple, List

logger = logging.getLogger(__name__)


class FileManager:
    """Менеджер для работы с файлами (сохранение/загрузка pickle)."""
    
    @staticmethod
    def save_pickle(data: Any, filepath: Path) -> None:
        """
        Сохраняет данные в pickle файл.
        
        Args:
            data: Данные для сохранения
            filepath: Путь к файлу
            
        Raises:
            IOError: Если не удается сохранить файл
        """
        try:
            # Создаем директорию если она не существует
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, "wb") as file:
                pickle.dump(data, file)
        except Exception as e:
            logger.error(f"Ошибка при сохранении файла {filepath}: {e}", exc_info=True)
            raise

    @staticmethod
    def load_pickle(filepath: Path) -> Any:
        """
        Загружает данные из pickle файла.
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            Загруженные данные
            
        Raises:
            FileNotFoundError: Если файл не найден
            IOError: Если не удается загрузить файл
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        try:
            with open(filepath, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {filepath}: {e}", exc_info=True)
            raise

    @staticmethod
    def get_source_filename(source_id: Tuple[int]) -> str:
        """
        Преобразует source_id в имя файла.
        
        Args:
            source_id: Кортеж с ID источника
            
        Returns:
            Имя файла без расширения
        """
        return (
            str(source_id)
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
            .replace(" ", "_")
        )
    
    @staticmethod
    def ensure_directory(path: Path) -> None:
        """
        Создает директорию если она не существует.
        
        Args:
            path: Путь к директории
        """
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def list_result_files(results_dir: Path, pattern: str = "*.pkl") -> List[Path]:
        """
        Получает список файлов результатов.
        
        Args:
            results_dir: Директория с результатами
            pattern: Паттерн для поиска файлов (по умолчанию "*.pkl")
            
        Returns:
            Список путей к файлам
        """
        if not results_dir.exists():
            logger.warning(f"Директория не найдена: {results_dir}")
            return []
        
        return list(results_dir.glob(pattern))
