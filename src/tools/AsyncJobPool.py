#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import asyncio
import time
import random
import traceback
import logging
from src.tracking.resource_tracking import MaxLLMCallsExceeded

logger = logging.getLogger(__name__)

class AsyncJobPool:
    """
    A pool to manage and run asyncio jobs with a short-circuiting condition.
    """
    def __init__(self, max_concurrent=None):
        self.tasks = []
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    def _is_failure(self, result):
        """
        Determine if a result represents a failure.
        
        Args:
            result: The task result to check
            
        Returns:
            bool: True if the result represents a failure
        """
        if isinstance(result, tuple) and len(result) >= 1:
            # If it's a tuple, check the first element
            return not result[0]
        else:
            # For non-tuple results, use truthiness
            return not result

    async def _semaphore_limited_task(self, coro, *args, **kwargs):
        """
        Wrapper that limits concurrency using semaphore if enabled.
        """
        if self.semaphore:
            async with self.semaphore:
                return await coro(*args, **kwargs)
        else:
            return await coro(*args, **kwargs)

    def submit(self, coro, *args, **kwargs):
        """
        Submits a coroutine to be run as a job.
        This method MUST wrap the coroutine in asyncio.create_task.
        """
        name = kwargs.pop('name', None)
        
        if self.semaphore:
            # Wrap the coroutine with semaphore limiting
            limited_coro = self._semaphore_limited_task(coro, *args, **kwargs)
            task = asyncio.create_task(limited_coro, name=name)
        else:
            # Original behavior - no concurrency limiting
            task = asyncio.create_task(coro(*args, **kwargs), name=name)
            
        self.tasks.append(task)
        if name:
            logger.info(f"📥 Submitted job: {name}")
        else:
            logger.info(f"📥 Submitted job: {task.get_name()}")

    async def wait_for_first_truthy(self):
        """
        Waits for jobs to complete, returning the first truthy result.
        This implementation correctly uses asyncio.wait to get completed tasks.
        """
        if not self.tasks:
            return None

        # We start with all tasks pending.
        pending_tasks = set(self.tasks)
        
        try:
            while pending_tasks:
                # Use asyncio.wait with FIRST_COMPLETED.
                # This waits until at least one task is done and returns two sets:
                # 'done' contains the completed Task objects.
                # 'pending' contains the tasks that are still running.
                done, pending = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Process the tasks that just finished.
                for completed_task in done:
                    job_name = completed_task.get_name()
                    try:
                        # Awaiting a completed task gets its result or raises its exception.
                        result = await completed_task
                        logger.info(f"✅ Job '{job_name}' finished")

                        if result:
                            logger.info(f"🏆 Truthy result found from '{job_name}'.")
                            logger.info("🛑 Stopping all other jobs...")
                            
                            # Cancel all remaining pending tasks.
                            for task in pending:
                                task.cancel()
                            
                            return result

                    except asyncio.CancelledError:
                        # This can happen if another task found a result and we were cancelled.
                        logger.info(f"❌ Job '{job_name}' was successfully cancelled.")
                    except MaxLLMCallsExceeded as e:
                        logger.info(f"🔥 Job '{job_name}' failed due to rate limit: {e}")
                        # Stop all other tasks
                        for task in pending:
                            task.cancel()
                        # Raise exception to stop further processing
                        raise e
                    except Exception as e:
                        logger.info(f"🔥 Job '{job_name}' failed with an exception: {e}")
                        traceback.print_exc()

                # For the next iteration, we only wait on the tasks that are still pending.
                pending_tasks = pending

            logger.info("\nAll jobs completed, no truthy result found.")
            return None

        finally:
            # Ensure all tasks are cancelled on exit, just in case.
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            self.tasks.clear()

    async def wait_for_all(self):
        """
        Waits for ALL jobs to complete and returns a list of all results.
        
        Concurrency is controlled by the semaphore set in __init__.
        If max_concurrent was specified, only that many jobs will run simultaneously.
        
        Returns:
            List of tuples: [(task_name, result_or_exception), ...]
            Where result_or_exception is either the task result or the exception that was raised.
        """
        if not self.tasks:
            return []

        results = []
        
        try:
            # Wait for all tasks to complete
            completed_results = await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Process all results
            for task, result in zip(self.tasks, completed_results):
                job_name = task.get_name()
                
                if isinstance(result, Exception):
                    if isinstance(result, asyncio.CancelledError):
                        logger.info(f"❌ Job '{job_name}' was cancelled.")
                    else:
                        logger.info(f"🔥 Job '{job_name}' failed with exception: {result}")
                    results.append((job_name, result))
                else:
                    logger.info(f"✅ Job '{job_name}' finished with result: {result}")
                    results.append((job_name, result))
                    
            logger.info(f"\n🏁 All {len(self.tasks)} jobs completed.")
            return results

        finally:
            # Clean up
            self.tasks.clear()

    async def wait_for_all_successful(self):
        """
        Waits for ALL jobs to complete and returns only the successful results.
        
        Returns:
            List of tuples: [(task_name, result), ...] for successful tasks only
        """
        all_results = await self.wait_for_all()
        
        # Filter for successful results (not exceptions)
        successful_results = [
            (name, result) for name, result in all_results
            if not isinstance(result, Exception)
        ]
        
        success_count = len(successful_results)
        total_count = len(all_results)
        logger.info(f"📊 {success_count}/{total_count} jobs completed successfully.")
        
        return successful_results

    async def wait_until_first_failure_or_all_success(self):
        """
        Waits for the first failure. If no failures, wait for all successes.
        
        Returns:
            If a failure occurs: tuple of (task_name, result_or_exception) for the failed task
            If all succeed: list of tuples [(task_name, result), ...] for all successful tasks
        """
        if not self.tasks:
            return []

        # We start with all tasks pending.
        pending_tasks = set(self.tasks)
        successful_results = []
        
        try:
            while pending_tasks:
                # Use asyncio.wait with FIRST_COMPLETED.
                # This waits until at least one task is done and returns two sets:
                # 'done' contains the completed Task objects.
                # 'pending' contains the tasks that are still running.
                done, pending = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Process the tasks that just finished.
                for completed_task in done:
                    job_name = completed_task.get_name()
                    try:
                        # Awaiting a completed task gets its result or raises its exception.
                        result = await completed_task
                        logger.info(f"✅ Job '{job_name}' finished with result: {result}")

                        # Check if this is a failure using the existing _is_failure method
                        if self._is_failure(result):
                            logger.info(f"💥 First failure found from '{job_name}': {result}")
                            logger.info("🛑 Stopping all other jobs...")
                            
                            # Cancel all remaining pending tasks.
                            for task in pending:
                                task.cancel()
                            
                            return [(job_name, result)]
                        else:
                            # This is a success, store it
                            successful_results.append((job_name, result))
                            logger.info(f"🏆 Success from '{job_name}': {result}")

                    except asyncio.CancelledError:
                        # This can happen if another task failed and we were cancelled.
                        logger.info(f"❌ Job '{job_name}' was successfully cancelled.")
                    
                    except MaxLLMCallsExceeded as e:
                        logger.error(f"🔥 Job '{job_name}' failed due to rate limit: {e}")
                        # Stop all other tasks
                        for task in pending:
                            task.cancel()
                        # Raise exception to stop further processing
                        raise e
                    
                    except Exception as e:
                        logger.info(f"💥 First failure found from '{job_name}': Exception {e}")
                        traceback.print_exc()
                        logger.info("🛑 Stopping all other jobs...")
                        
                        # Cancel all remaining pending tasks.
                        for task in pending:
                            task.cancel()
                        
                        return [(job_name, e)]

                # For the next iteration, we only wait on the tasks that are still pending.
                pending_tasks = pending

            logger.info(f"\n🎉 All {len(successful_results)} jobs completed successfully!")
            return successful_results

        finally:
            # Ensure all tasks are cancelled on exit, just in case.
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            self.tasks.clear()

    async def wait_for_first_failure_after_successes(self):
        """
        Waits for the first failure (false result) that was preceded by all successful (true) results
        in SUBMISSION order.
        
        For example, if jobs are submitted as J1, J2, J3, J4 and complete as J1, J2, J4, J3:
        - J1(true), J2(true), J4(false) complete first
        - When J3(false) completes, we check if J1, J2 (submitted before J3) both succeeded
        - Since they did, we return J3 as the first failure after all previous successes
        
        Returns:
            tuple: (failed_task_name, all_results_in_submission_order) if such a failure is found, 
                   None if no such pattern exists
            all_results_in_submission_order is a list of (task_name, result) tuples in submission order
        """
        if not self.tasks:
            return None
        
        # We start with all tasks pending
        pending_tasks = set(self.tasks)
        task_results = {}  # Maps task -> result
        
        try:
            while pending_tasks:
                # Wait for at least one task to complete
                done, pending = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process the tasks that just finished
                for completed_task in done:
                    job_name = completed_task.get_name()
                    try:
                        # Get the result of the completed task
                        result = await completed_task
                        logger.info(f"✅ Job '{job_name}' finished with result: {result}")
                        
                        # Store the result
                        task_results[completed_task] = result
                        
                        # Check for first failure
                        all_previous_completed_and_successful = True
                        failed_task = None
                        for curr_task in self.tasks:
                            if curr_task not in task_results:
                                # Previous task hasn't completed yet
                                all_previous_completed_and_successful = False
                                break
                            elif self._is_failure(task_results[curr_task]):
                                failed_task = curr_task
                                break
                        
                        if all_previous_completed_and_successful and failed_task:
                            failed_job_name = failed_task.get_name()
                            logger.info(f"🎯 Found first failure '{failed_job_name}' after all previous successes in submission order!")
                            logger.info("🛑 Stopping all other jobs...")
                            
                            # Cancel all remaining pending tasks
                            for task in pending:
                                task.cancel()
                            
                            # Build results for all completed tasks
                            results_in_submission_order = []
                            for task in self.tasks:
                                if task in task_results:
                                    task_name = task.get_name()
                                    result = task_results[task]
                                    results_in_submission_order.append((task_name, result))
                            
                            return (failed_job_name, results_in_submission_order)
                            
                    except asyncio.CancelledError:
                        logger.info(f"❌ Job '{job_name}' was cancelled.")

                    except MaxLLMCallsExceeded as e:
                        logger.info(f"🔥 Job '{job_name}' failed due to rate limit: {e}")
                        # Stop all other tasks
                        for task in pending:
                            task.cancel()
                        # Raise exception to stop further processing
                        raise e

                    except Exception as e:
                        logger.info(f"🔥 Job '{job_name}' failed with exception: {e}")
                        traceback.print_exc()
                        # Treat exceptions as failures for this logic
                        task_results[completed_task] = False
                
                # Update pending tasks for next iteration
                pending_tasks = pending
            
            logger.info("\n🏁 All jobs completed - no failure found after all successes.")
            
            # Return all results since everything succeeded  
            results_in_submission_order = []
            for task in self.tasks:
                if task in task_results:
                    task_name = task.get_name() 
                    result = task_results[task]
                    results_in_submission_order.append((task_name, result))
            
            return (None, results_in_submission_order)
            
        finally:
            # Ensure all tasks are cancelled on exit
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            self.tasks.clear()
