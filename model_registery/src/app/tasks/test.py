from celery import shared_task
from ..crud.logger import AppLogger


logger = AppLogger(__name__).get_logger()
@shared_task(name='test_task')
def test_task(param1, param2):
    logger.info(f'Processing {param1} and {param2}')
    result = param1 + param2
    logger.info(f'Result: {result}')
    return result