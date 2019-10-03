/***************************************************************************
 *
 * Authors:    Jan Polák (456647@mail.muni.cz)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include "reconstruct_fourier_scheduler.h"
#include "reconstruct_fourier_codelets.h"
#include <starpu.h>
#include <cassert>
#include <cstdio>
#include <cinttypes>
#include <atomic>
#include <algorithm>

#if RFS_LOGGING
// FOR DEBUGING (pid_t x = syscall(__NR_gettid);)
#include <sys/types.h>
#include <sys/syscall.h>
#define RFS_LOG(message, ...) do{\
	FILE* file = ((rfs_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id))->log;\
    fprintf(file, "[%s : %zu : %d] " message "\n", __FUNCTION__, (size_t) syscall(SYS_gettid), sched_ctx_id, ##__VA_ARGS__);\
    fflush(file);\
    } while(0)
#else
#define RFS_LOG(message, ...) ((void)(message))
#endif

void task_data::recompute_metrics(const std::bitset<STARPU_NMAXWORKERS> &available_workers, unsigned sched_ctx_id) {
	for (int workerid = 0; workerid < STARPU_NMAXWORKERS; ++workerid) {
		best_implementation_by_worker[workerid] = -1;
		best_implementation_time_by_worker[workerid] = MAX_TIME;

		if (!available_workers[workerid]) {
			continue;
		}

		unsigned mask;
		if (!starpu_worker_can_execute_task_impl(workerid, task, &mask)) {
			continue;
		}

		unsigned pickedImpl = 0;
		uint64_t pickedImplTime = std::numeric_limits<uint64_t>::max();
		const auto perfArchetype = starpu_worker_get_perf_archtype(workerid, sched_ctx_id);

		for (unsigned impl = 0; impl < sizeof(unsigned) && mask != 0; ++impl) {
			unsigned bit = 1u << impl;
			if (mask & bit) {
				mask &= ~bit;

				uint64_t implTime = sanitize_time(starpu_task_expected_data_transfer_time_for(task, workerid))
						+ sanitize_time(starpu_task_expected_length(task, perfArchetype, impl));

				if (implTime < pickedImplTime) {
					pickedImpl = impl;
					pickedImplTime = implTime;
				}
			}
		}

		best_implementation_by_worker[workerid] = pickedImpl;
		best_implementation_time_by_worker[workerid] = pickedImplTime;
	}
}

int task_data::find_best_worker(const std::bitset<STARPU_NMAXWORKERS> &available_workers) {
	int pickedWorkerIds[STARPU_NMAXWORKERS];
	unsigned pickedWorkerIdsCount = 0;
	double pickedWorkerIdFinishTime = INFINITY;

	for (int workerId = 0; workerId < STARPU_NMAXWORKERS; ++workerId) {
		if (!available_workers[workerId] || best_implementation_by_worker[workerId] == -1) {
			continue;
		}

		double finishTime = best_implementation_time_by_worker[workerId];
		if (finishTime < pickedWorkerIdFinishTime) {
			pickedWorkerIds[0] = workerId;
			pickedWorkerIdsCount = 1;
			pickedWorkerIdFinishTime = finishTime;
		} else if (finishTime == pickedWorkerIdFinishTime) {
			pickedWorkerIds[pickedWorkerIdsCount++] = workerId;
		}
	}

	assert(pickedWorkerIdsCount > 0 && "No worker to schedule task to");

	// It is not very important what this value is, it just should be somewhat unique for each run, so that most tasks
	// are uniformly distributed across available workers.
	static std::atomic<unsigned> roundRobinNext { 0 };
	unsigned pickedWorkerIndex = roundRobinNext++ % pickedWorkerIdsCount;

	return pickedWorkerIds[pickedWorkerIndex];
}

static void rfs_init_sched(unsigned sched_ctx_id) {
	auto* data = new rfs_data;
	assert(data);
#if RFS_LOGGING
	data->log = fopen("rec_fou_scheduler.log", "wa");
	assert(data->log);
#endif
	starpu_sched_ctx_set_policy_data(sched_ctx_id, data);

	data->timing.load("rfs_timings.txt");

	RFS_LOG("");
}

static void rfs_deinit_sched(unsigned sched_ctx_id) {
	RFS_LOG("");

	auto* data = (rfs_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	assert(data);
#if RFS_LOGGING
	fclose(data->log);
#endif

	data->timing.save("rfs_timings.txt");

	delete data;
	starpu_sched_ctx_set_policy_data(sched_ctx_id, nullptr);
}

static void rfs_add_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers) {
	auto* data = (rfs_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	for (unsigned i = 0; i < nworkers; ++i) {
		const int worker = workerids[i];
		if (starpu_worker_is_combined_worker(worker)) {
			continue;// Just in case
		}
		data->workers_mask.set(worker);
		data->workers[worker].deviceKey = timing_device_key(starpu_worker_get_type(worker), starpu_worker_get_devid(worker));
		data->workers[worker].busy = true;
		RFS_LOG("%d", worker);
	}

	// Recreate next-worker cycles
	for (int worker = 0; worker < STARPU_NMAXWORKERS; ++worker) {
		if (!data->workers_mask[worker])
			continue;

		// (reset captains for next step)
		data->workers[worker].combinedWorkerCaptain = -1;

		data->workers[worker].nextSimilarWorker = worker;
		for (int nextWorkerOffset = 1; nextWorkerOffset < STARPU_NMAXWORKERS; ++nextWorkerOffset) {
			int nextWorker = (worker + nextWorkerOffset) % STARPU_NMAXWORKERS;
			if (!data->workers_mask[nextWorker] || data->workers[worker].deviceKey != data->workers[nextWorker].deviceKey)
				continue;
			data->workers[worker].nextSimilarWorker = nextWorker;
			break;
		}
	}

	// Find combined workers
	const unsigned workerCount = starpu_worker_get_count();
	const unsigned combinedWorkerCount = starpu_combined_worker_get_count();
	for (int combinedWorker = (int) workerCount; combinedWorker < workerCount + combinedWorkerCount; ++combinedWorker) {
		int size = 0;
		int* workers = nullptr;
		starpu_combined_worker_get_description(combinedWorker, &size, &workers);
		if (size <= 1) {
			continue;
		}

		bool valid = true;
		starpu_worker_archtype combinedType = starpu_worker_get_type(workers[0]);
		for (int i = 0; i < size; ++i) {
			if (!data->workers_mask[workers[i]]) {
				valid = false;
				break;
			}
			auto type = starpu_worker_get_type(workers[i]);
			if (type != combinedType) {
				RFS_LOG("Mixed type worker not supported %d", combinedWorker);
				valid = false;
				break;
			}
		}

		if (!valid) {
			continue;
		}

		// We can use this combined worker!
		bool used = false;
		for (int i = 0; i < size; ++i) {
			auto& wData = data->workers[workers[i]];
			if (wData.combinedWorkerCaptain == -1) {
				wData.combinedWorkerCaptain = combinedWorker;
				used = true;
				break;
			}
		}

		if (!used) {
			RFS_LOG("Can't use combined worker %d, no available captain", combinedWorker);
			continue;
		}

		RFS_LOG("Registered combined worker %d (%d workers)", combinedWorker, size);
	}
}

static void rfs_remove_workers(unsigned sched_ctx_id, int *workerids, unsigned nworkers) {
	auto* data = (rfs_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	for (unsigned i = 0; i < nworkers; ++i) {
		const int worker = workerids[i];
		data->workers_mask.reset(worker);
		RFS_LOG("%d (load error: %" PRIu64 ")", worker, data->workers[worker].queued_load);
	}
}

static int rfs_push_task(starpu_task * task) {
	const unsigned sched_ctx_id = task->sched_ctx;
	//RFS_LOG("%s (%s)", task->name, task->cl->name);

	auto* data = (rfs_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);

	task_data taskData(task);
	taskData.recompute_metrics(data->workers_mask, sched_ctx_id);

	// Find best worker
	int pickedWorkerId = taskData.find_best_worker(data->workers_mask);

	starpu_task_set_implementation(task, taskData.best_implementation_by_worker[pickedWorkerId]);

	starpu_worker_lock(pickedWorkerId);
	worker_data& pickedWorkerData = data->workers[pickedWorkerId];

	pickedWorkerData.workerMutex.lock();
	pickedWorkerData.queue.addLast(taskData);
	pickedWorkerData.queued_load += taskData.best_implementation_time_by_worker[pickedWorkerId];
#if RFS_LOGGING
	assert(pickedWorkerData.isQueuedLoadCorrect(pickedWorkerId));
#endif
	uint64_t finishTime = pickedWorkerData.current_load + pickedWorkerData.queued_load;
	pickedWorkerData.workerMutex.unlock();

	starpu_push_task_end(task);
	starpu_idle_prefetch_task_input_for(task, pickedWorkerId);
	starpu_wake_worker_locked(pickedWorkerId);
	starpu_worker_unlock(pickedWorkerId);


	RFS_LOG("Scheduled onto %d (%" PRIu64 " us)", pickedWorkerId, finishTime);
	return 0;
}

static void rfs_push_task_notify(STARPU_ATTRIBUTE_UNUSED starpu_task * task, int workerid, STARPU_ATTRIBUTE_UNUSED int perf_workerid, unsigned sched_ctx_id) {
	//RFS_LOG("%s (%s) %d %d", task->name, task->cl->name, workerid, perf_workerid);
	auto* data = (rfs_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	data->workers[workerid].busy = true;
}

static starpu_task* steal_single_parallel_job(unsigned thiefWorkerId, unsigned combinedWorkerThiefId, unsigned thiefWorkerGroupSize,
                                              unsigned victimWorkerId, worker_data &victimWorkerData,
                                              uint32_t &workerJobLength) {
	uint64_t& victimWorkerLoad = victimWorkerData.queued_load;
	starpu_task* stolenTask = nullptr;
	rfs::removeMatching(victimWorkerData.queue, [thiefWorkerId, combinedWorkerThiefId, thiefWorkerGroupSize, victimWorkerId, &victimWorkerLoad, &stolenTask, &workerJobLength](task_data& task){
		if (task.task->cl == nullptr) {
			// Synchronization task, better not
			return rfs::RemoveResult::KeepContinue;
		}

		if (task.best_implementation_by_worker[thiefWorkerId] == -1) {
			// Can't run this, can't steal
			return rfs::RemoveResult::KeepContinue;
		}

		if (task.task->cl->type == starpu_codelet_type::STARPU_SEQ) {
			// We want only parallel tasks
			return rfs::RemoveResult::KeepContinue;
		}

		if (task.task->cl->max_parallelism < thiefWorkerGroupSize) {
			// This thief is too big for this job
			return rfs::RemoveResult::KeepContinue;
		}

		uint32_t totalThiefLoadAfterSteal = task.best_implementation_time_by_worker[thiefWorkerId] / thiefWorkerGroupSize;
		uint64_t totalVictimLoadAfterSteal = victimWorkerLoad - task.best_implementation_time_by_worker[victimWorkerId];
		if (totalThiefLoadAfterSteal >= totalVictimLoadAfterSteal) {
			// We can steal it, but it won't be helpful
			return rfs::RemoveResult::KeepContinue;
		}

		if (!starpu_combined_worker_can_execute_task(combinedWorkerThiefId, task.task, task.best_implementation_by_worker[thiefWorkerId])) {
			// But it turns out, that we can't run it
			return rfs::RemoveResult::KeepContinue;
		}

		// It's payday fellas
		workerJobLength = totalThiefLoadAfterSteal;
		starpu_task_set_implementation(task.task, task.best_implementation_by_worker[thiefWorkerId]);
		victimWorkerLoad = totalVictimLoadAfterSteal;
		stolenTask = task.task;
		return rfs::RemoveResult::RemoveBreak;
	});

	return stolenTask;
}

static unsigned steal_jobs_until_equilibrium(int thiefWorkerId, int victimWorkerId, worker_data &thiefWorkerData, worker_data &victimWorkerData) {
	auto thiefIsCPU = starpu_worker_get_type(thiefWorkerId) == starpu_worker_archtype::STARPU_CPU_WORKER;
	unsigned stolen = 0;

	uint64_t& victimWorkerLoad = victimWorkerData.queued_load;
	rfs::removeMatching(victimWorkerData.queue, [&thiefWorkerData, thiefWorkerId, victimWorkerId, &victimWorkerLoad, &stolen, thiefIsCPU](task_data& task){
		if (task.best_implementation_by_worker[thiefWorkerId] == -1) {
			// Can't steal this
			return rfs::RemoveResult::KeepContinue;
		}

		if (thiefIsCPU && task.task->cl->type != starpu_codelet_type::STARPU_SEQ) {
			// CPU's should not steal tasks that are parallel on CPU - not yet
			return rfs::RemoveResult::KeepContinue;
		}

		uint64_t totalThiefLoadAfterSteal = thiefWorkerData.queued_load + task.best_implementation_time_by_worker[thiefWorkerId];
		//assert(victimWorkerData.queued_load >= task.best_implementation_time_by_worker[victimWorkerId]);
		uint64_t totalVictimLoadAfterSteal = victimWorkerLoad - task.best_implementation_time_by_worker[victimWorkerId];
		if (totalThiefLoadAfterSteal >= totalVictimLoadAfterSteal) {
			// We can steal it, but it won't be helpful
			return rfs::RemoveResult::KeepContinue;
		}

		// It's payday fellas
		starpu_task_set_implementation(task.task, task.best_implementation_by_worker[thiefWorkerId]);
		thiefWorkerData.queue.addLast(task);
		thiefWorkerData.queued_load = totalThiefLoadAfterSteal;
		victimWorkerLoad = totalVictimLoadAfterSteal;
		stolen++;

		return rfs::RemoveResult::RemoveContinue;
	});

#if RFS_LOGGING
	assert(thiefWorkerData.isQueuedLoadCorrect(thiefWorkerId));
	assert(victimWorkerData.isQueuedLoadCorrect(victimWorkerId));
#endif

#if RFS_LOGGING
	if (stolen > 0)
		fprintf(stderr, "%d steals from %d: %d tasks\n", thiefWorkerId, victimWorkerId, stolen);
#endif

	return stolen;
}

static starpu_task * rfs_pop_task(unsigned sched_ctx_id) {
	auto* data = (rfs_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	int workerId = starpu_worker_get_id_check();
	assert(!starpu_worker_is_combined_worker(workerId));

	auto& ourWorkerData = data->workers[workerId];
	std::lock_guard<std::mutex> lock(ourWorkerData.workerMutex);

	ourWorkerData.current_load = 0;

	if (ourWorkerData.queue.empty()) {
		assert(ourWorkerData.queued_load == 0);
		// Fill up the queue first

		// Tracking which workers were already tried to not bother them again.
		// Considering all workers that are not present as already checked, to simplify logic down the line.
		std::bitset<STARPU_NMAXWORKERS> checked = ~data->workers_mask;
		checked[workerId] = true; // Don't check self

		// 1. Steal from similar workers

		int nextSimilarWorkerId = ourWorkerData.nextSimilarWorker;
		while (nextSimilarWorkerId != workerId /* this is cyclic */) {
			auto& nextSimilarWorkerData = data->workers[nextSimilarWorkerId];

			// Only try_lock instead of hard lock to prevent deadlocks
			if (nextSimilarWorkerData.workerMutex.try_lock()) {
				checked[nextSimilarWorkerId] = true;
				steal_jobs_until_equilibrium(workerId, nextSimilarWorkerId, ourWorkerData, nextSimilarWorkerData);
				nextSimilarWorkerData.workerMutex.unlock();
			}

			if (!ourWorkerData.queue.empty()) {
				goto stealingDone;
			}

			nextSimilarWorkerId = nextSimilarWorkerData.nextSimilarWorker;
		}

		// 2. Steal from all workers

		// Using own id as offset to distribute stealing in a round robin fashion
		for (int stealI = 0; stealI < STARPU_NMAXWORKERS; ++stealI) {
			const int victimWorker = (workerId + stealI) % STARPU_NMAXWORKERS;
			if (checked[victimWorker]) {
				continue;
			}

			auto &victimWorkerData = data->workers[victimWorker];
			if (victimWorkerData.workerMutex.try_lock()) {
				steal_jobs_until_equilibrium(workerId, victimWorker, ourWorkerData, victimWorkerData);
				victimWorkerData.workerMutex.unlock();

				if (!ourWorkerData.queue.empty()) {
					goto stealingDone;
				}
			} // else: Lock failed, worker busy, spare them
		}

		// 3. If combined worker captain and no sub-workers are busy, try to steal something combined.
		if (ourWorkerData.combinedWorkerCaptain >= 0) {
			// We are a captain! See if companions are free to do some work.
			int combinedWorkerThiefId = ourWorkerData.combinedWorkerCaptain;
			int groupSize = 0;
			int* groupMembers = nullptr;
			starpu_combined_worker_get_description(combinedWorkerThiefId, &groupSize, &groupMembers);

			bool allFree = true;
			for (int i = 0; i < groupSize; ++i) {
				if (groupMembers[i] != workerId && data->workers[groupMembers[i]].busy) {
					allFree = false;
					break;
				}
			}

			if (allFree) {
				// There is a real chance of a parallel race, but lets not think about that for now.

				// Pick a task to steal!
				uint32_t stolenJobPerWorkerLength = 0;
				starpu_task* stolenJob = nullptr;
				// Same worker-checking logic as case 2
				for (int stealI = 0; stealI < STARPU_NMAXWORKERS; ++stealI) {
					const int victimWorker = (workerId + stealI) % STARPU_NMAXWORKERS;
					if (checked[victimWorker]) {
						continue;
					}

					auto &victimWorkerData = data->workers[victimWorker];
					if (victimWorkerData.workerMutex.try_lock()) {
						stolenJob = steal_single_parallel_job(workerId, combinedWorkerThiefId, groupSize, victimWorker, victimWorkerData,
						                                      stolenJobPerWorkerLength);
						victimWorkerData.workerMutex.unlock();

						break;
					} // else: Lock failed, worker busy, spare them
				}

				if (stolenJob != nullptr) {
					// Got something, use it!
					starpu_parallel_task_barrier_init(stolenJob, combinedWorkerThiefId);

					for (int i = 0; i < groupSize; i++) {
						const int groupMember = groupMembers[i];
						data->workers[groupMember].busy = true;
						// Not setting current_load in case of race condition, in which case it would be wrong.
						// But not setting it up is probably not a big deal.

						if (groupMember == workerId) {
							continue;
						}

						struct starpu_task *alias = starpu_task_dup(stolenJob);
						alias->destroy = 1;
						starpu_push_local_task(groupMembers[i], alias, 1 /* take this first */);
					}

					// Not sure wtf, but both StarPU parallel schedulers do it
					struct starpu_task *masterAlias = starpu_task_dup(stolenJob);
					masterAlias->destroy = 1;

					ourWorkerData.current_load = stolenJobPerWorkerLength;
					fprintf(stderr, "Stolen %s and distributed it over %d workers\n", stolenJob->cl->name, groupSize);
					return masterAlias;
				}
			}
		}
	}

	stealingDone:

	if (ourWorkerData.queue.empty()) {
		// And we still didn't find anything
		data->workers[workerId].busy = false;
		return nullptr;
	}

	// Normal operation, taking tasks from own queue
	auto& taskData = ourWorkerData.queue.first();
	starpu_task* task = taskData.task;
	auto taskTime = taskData.best_implementation_time_by_worker[workerId];
	ourWorkerData.queued_load -= taskTime;
	ourWorkerData.current_load = taskTime;
	ourWorkerData.queue.removeFirst();

#if RFS_LOGGING
	assert(ourWorkerData.isQueuedLoadCorrect(workerId));
#endif
	data->workers[workerId].busy = true;
	return task;
}

static void rfs_pre_exec_hook(starpu_task * task, unsigned sched_ctx_id) {
	//RFS_LOG("%s (%s)", task->name, task->cl->name);
	if (!task->cl) {
		return;
	}

	auto* data = (rfs_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	data->workers[starpu_worker_get_id_check()].execStartTimeUs = starpu_timing_now();
}

static void rfs_post_exec_hook(starpu_task * task, unsigned sched_ctx_id) {
	//RFS_LOG("%s (%s)", task->name, task->cl->name);
	if (!task->cl) {
		return;
	}

	auto* data = (rfs_data*)starpu_sched_ctx_get_policy_data(sched_ctx_id);
	auto workerId = starpu_worker_get_id_check();
	auto& workerData = data->workers[workerId];

	// Measure task time
	double timeElapsed = starpu_timing_now() - workerData.execStartTimeUs;
	unsigned impl = starpu_task_get_implementation(task);

	size_t size = 0;
	if (task->cl->model && task->cl->model->size_base) {
		size = task->cl->model->size_base(task, impl);
	}

	data->timing.record(task->cl, impl, starpu_worker_get_type(workerId), starpu_worker_get_devid(workerId), size, timeElapsed);
}

Schedulers::Schedulers() noexcept : reconstruct_fourier{} {
	reconstruct_fourier.init_sched = rfs_init_sched;
	reconstruct_fourier.deinit_sched = rfs_deinit_sched;
	reconstruct_fourier.push_task = rfs_push_task;
	reconstruct_fourier.push_task_notify = rfs_push_task_notify;
	reconstruct_fourier.pop_task = rfs_pop_task;
	reconstruct_fourier.pre_exec_hook = rfs_pre_exec_hook;
	reconstruct_fourier.post_exec_hook = rfs_post_exec_hook;
	reconstruct_fourier.add_workers = rfs_add_workers;
	reconstruct_fourier.remove_workers = rfs_remove_workers;
	reconstruct_fourier.policy_name = "reconstruct_fourier";
	reconstruct_fourier.policy_description = "Specialized scheduler for Fourier reconstruction";
}

Schedulers schedulers;
