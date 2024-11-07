from typing import List

from fastapi import APIRouter, Depends, status
from sqlmodel.ext.asyncio.session import AsyncSession

from src.auth.dependencies import RoleChecker
from src.books.schemas import Book
from src.db.main import get_session
from .schemas import TagAddModel, TagCreateModel, TagModel
from .service import TagService

router = APIRouter(prefix="/tags", tags=["tags"])
tag_service = TagService()
require_user = Depends(RoleChecker(["user", "admin"]))


@router.get(
    "/",
    response_model=List[TagModel],
    dependencies=[require_user],
    summary="Get all tags"
)
async def get_all_tags(
    session: AsyncSession = Depends(get_session)
) -> List[TagModel]:
    """Retrieve all tags."""
    return await tag_service.get_tags(session)


@router.post(
    "/",
    response_model=TagModel,
    status_code=status.HTTP_201_CREATED,
    dependencies=[require_user],
    summary="Create new tag"
)
async def create_tag(
    tag_data: TagCreateModel,
    session: AsyncSession = Depends(get_session)
) -> TagModel:
    """Create a new tag."""
    return await tag_service.add_tag(tag_data=tag_data, session=session)


@router.post(
    "/books/{book_uid}/tags",
    response_model=Book,
    dependencies=[require_user],
    summary="Add tags to book"
)
async def add_tags_to_book(
    book_uid: str,
    tag_data: TagAddModel,
    session: AsyncSession = Depends(get_session)
) -> Book:
    """Add tags to an existing book."""
    return await tag_service.add_tags_to_book(
        book_uid=book_uid,
        tag_data=tag_data,
        session=session
    )


@router.put(
    "/{tag_uid}",
    response_model=TagModel,
    dependencies=[require_user],
    summary="Update tag"
)
async def update_tag(
    tag_uid: str,
    tag_update_data: TagCreateModel,
    session: AsyncSession = Depends(get_session)
) -> TagModel:
    """Update an existing tag."""
    return await tag_service.update_tag(tag_uid, tag_update_data, session)


@router.delete(
    "/{tag_uid}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[require_user],
    summary="Delete tag"
)
async def delete_tag(
    tag_uid: str,
    session: AsyncSession = Depends(get_session)
) -> None:
    """Delete an existing tag."""
    await tag_service.delete_tag(tag_uid, session)
    return None