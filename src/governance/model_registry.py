from __future__ import annotations

from typing import List, Optional

from sqlmodel import select

from src.db import session_scope
from src.models import ModelPromotionEvent, ModelRegistryEntry


class ModelRegistryService:
    def ensure_default_champion(self, *, model_version: str, model_path: str, calibrator_path: str) -> None:
        with session_scope() as session:
            existing = session.exec(
                select(ModelRegistryEntry).where(ModelRegistryEntry.model_version == model_version)
            ).first()
            if existing is None:
                session.add(
                    ModelRegistryEntry(
                        model_version=model_version,
                        model_path=model_path,
                        calibrator_path=calibrator_path,
                        is_champion=True,
                    )
                )
                return

            existing.model_path = model_path
            existing.calibrator_path = calibrator_path
            if not existing.is_champion:
                current_champ = session.exec(select(ModelRegistryEntry).where(ModelRegistryEntry.is_champion == True)).first()
                if current_champ is None:
                    existing.is_champion = True

    def list_entries(self) -> List[ModelRegistryEntry]:
        with session_scope() as session:
            return list(session.exec(select(ModelRegistryEntry).order_by(ModelRegistryEntry.created_at.desc())).all())

    def get_entry(self, model_version: str) -> Optional[ModelRegistryEntry]:
        with session_scope() as session:
            return session.exec(
                select(ModelRegistryEntry).where(ModelRegistryEntry.model_version == model_version)
            ).first()

    def get_champion(self) -> Optional[ModelRegistryEntry]:
        with session_scope() as session:
            return session.exec(select(ModelRegistryEntry).where(ModelRegistryEntry.is_champion == True)).first()

    def promote(self, *, new_model_version: str, reason: str) -> Optional[ModelPromotionEvent]:
        with session_scope() as session:
            new_model = session.exec(
                select(ModelRegistryEntry).where(ModelRegistryEntry.model_version == new_model_version)
            ).first()
            if new_model is None:
                return None

            current = session.exec(select(ModelRegistryEntry).where(ModelRegistryEntry.is_champion == True)).first()
            old_model_version = current.model_version if current else ""

            if current:
                current.is_champion = False
            new_model.is_champion = True

            event = ModelPromotionEvent(
                old_model=old_model_version,
                new_model=new_model.model_version,
                reason=reason,
            )
            session.add(event)
            session.flush()
            return event

    def promotions(self, limit: int = 100) -> List[ModelPromotionEvent]:
        with session_scope() as session:
            return list(session.exec(select(ModelPromotionEvent).order_by(ModelPromotionEvent.timestamp.desc()).limit(limit)).all())
