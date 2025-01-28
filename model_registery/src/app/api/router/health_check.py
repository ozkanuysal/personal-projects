import psutil
from fastapi import APIRouter, HTTPException, Depends

from ...crud.logger import AppLogger
from ...models.user import User
from ...services.auth import auth_service

health_check_router = APIRouter(tags=["Health"])

logger = AppLogger(__name__).get_logger()

health_check_router = APIRouter()


@health_check_router.get("/status", tags=["Status"])
async def health_check(current_user: User = Depends(auth_service.get_current_user)):
    """
    This function performs a health check of the application. It collects system resource usage information
    and returns it in a JSON format.

    Parameters:
    current_user (User): The currently authenticated user. This parameter is optional and is retrieved using the
                         auth_service.get_current_user function.

    Returns:
    dict: A dictionary containing the health check results. The dictionary has the following structure:
          {
              "status": str,  # The health status of the application.
              "memory": {
                  "total": int,  # Total memory in bytes.
                  "available": int,  # Available memory in bytes.
                  "percent": float,  # Percentage of memory used.
                  "used": int,  # Memory used in bytes.
                  "free": int  # Memory free in bytes.
              },
              "cpu_usage": str,  # CPU usage percentage.
              "disk_usage": {
                  "total": int,  # Total disk space in bytes.
                  "used": int,  # Disk space used in bytes.
                  "free": int,  # Disk space free in bytes.
                  "percent": float  # Percentage of disk space used.
              }
          }

    Raises:
    HTTPException: If an error occurs during the health check, a 500 Internal Server Error is raised with
                   a detailed error message.
    """
    try:
        logger.info("Health check initiated by user: %s", current_user)
        memory_info = psutil.virtual_memory()

        cpu_usage = psutil.cpu_percent(interval=1)

        disk_usage = psutil.disk_usage('/')

        return {
            "status": "healthy",
            "memory": {
                "total": memory_info.total,
                "available": memory_info.available,
                "percent": memory_info.percent,
                "used": memory_info.used,
                "free": memory_info.free
            },
            "cpu_usage": f"{cpu_usage}%",
            "disk_usage": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": disk_usage.percent
            },

        }

    except Exception as e:
        logger.error("Health check failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")