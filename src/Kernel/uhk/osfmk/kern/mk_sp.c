/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/*
 * @OSF_COPYRIGHT@
 *
 */

/* The routines in this module are all obsolete */

#include <mach/boolean.h>
#include <mach/thread_switch.h>
#include <ipc/ipc_port.h>
#include <ipc/ipc_space.h>
#include <kern/ipc_kobject.h>
#include <kern/processor.h>
#include <kern/sched.h>
#include <kern/sched_prim.h>
#include <kern/spl.h>
#include <kern/task.h>
#include <kern/thread.h>
#include <mach/policy.h>
#include <kern/policy_internal.h>

#include <kern/syscall_subr.h>
#include <mach/mach_host_server.h>
#include <mach/mach_syscalls.h>

#include <kern/misc_protos.h>
#include <kern/spl.h>
#include <kern/sched.h>
#include <kern/sched_prim.h>
#include <kern/assert.h>
#include <kern/thread.h>
#include <mach/mach_host_server.h>
#include <mach/thread_act_server.h>
#include <mach/host_priv_server.h>


/*
 *	thread_set_policy
 *
 *	Set scheduling policy and parameters, both base and limit, for
 *	the given thread. Policy can be any policy implemented by the
 *	processor set, whether enabled or not.
 */
kern_return_t
thread_set_policy(
	thread_t                                thread,
	processor_set_t                 pset,
	policy_t                                policy,
	policy_base_t                   base,
	mach_msg_type_number_t  base_count,
	policy_limit_t                  limit,
	mach_msg_type_number_t  limit_count)
{
	int                                     max, bas;
	kern_return_t                   result = KERN_SUCCESS;

	if (thread == THREAD_NULL ||
	    pset == PROCESSOR_SET_NULL || pset != &pset0) {
		return KERN_INVALID_ARGUMENT;
	}

	if (invalid_policy(policy)) {
		return KERN_INVALID_ARGUMENT;
	}

	switch (policy) {
	case POLICY_RR:
	{
		policy_rr_base_t                rr_base = (policy_rr_base_t) base;
		policy_rr_limit_t               rr_limit = (policy_rr_limit_t) limit;

		if (base_count != POLICY_RR_BASE_COUNT ||
		    limit_count != POLICY_RR_LIMIT_COUNT) {
			result = KERN_INVALID_ARGUMENT;
			break;
		}

		bas = rr_base->base_priority;
		max = rr_limit->max_priority;
		if (invalid_pri(bas) || invalid_pri(max)) {
			result = KERN_INVALID_ARGUMENT;
			break;
		}

		break;
	}

	case POLICY_FIFO:
	{
		policy_fifo_base_t              fifo_base = (policy_fifo_base_t) base;
		policy_fifo_limit_t             fifo_limit = (policy_fifo_limit_t) limit;

		if (base_count != POLICY_FIFO_BASE_COUNT ||
		    limit_count != POLICY_FIFO_LIMIT_COUNT) {
			result = KERN_INVALID_ARGUMENT;
			break;
		}

		bas = fifo_base->base_priority;
		max = fifo_limit->max_priority;
		if (invalid_pri(bas) || invalid_pri(max)) {
			result = KERN_INVALID_ARGUMENT;
			break;
		}

		break;
	}

	case POLICY_TIMESHARE:
	{
		policy_timeshare_base_t         ts_base = (policy_timeshare_base_t) base;
		policy_timeshare_limit_t        ts_limit =
		    (policy_timeshare_limit_t) limit;

		if (base_count != POLICY_TIMESHARE_BASE_COUNT ||
		    limit_count != POLICY_TIMESHARE_LIMIT_COUNT) {
			result = KERN_INVALID_ARGUMENT;
			break;
		}

		bas = ts_base->base_priority;
		max = ts_limit->max_priority;
		if (invalid_pri(bas) || invalid_pri(max)) {
			result = KERN_INVALID_ARGUMENT;
			break;
		}

		break;
	}

	default:
		result = KERN_INVALID_POLICY;
	}

	if (result != KERN_SUCCESS) {
		return result;
	}

	/* Note that we do not pass on max priority. */
	if (result == KERN_SUCCESS) {
		result = thread_set_mode_and_absolute_pri(thread, policy, bas);
	}

	return result;
}


/*
 *      thread_policy
 *
 *	Set scheduling policy and parameters, both base and limit, for
 *	the given thread. Policy must be a policy which is enabled for the
 *	processor set. Change contained threads if requested.
 */
kern_return_t
thread_policy(
	thread_t                                thread,
	policy_t                                policy,
	policy_base_t                   base,
	mach_msg_type_number_t  count,
	boolean_t                               set_limit)
{
	kern_return_t                   result = KERN_SUCCESS;
	processor_set_t                 pset = &pset0;
	policy_limit_t                  limit = NULL;
	int                                             limcount = 0;
	policy_rr_limit_data_t                  rr_limit;
	policy_fifo_limit_data_t                fifo_limit;
	policy_timeshare_limit_data_t   ts_limit;

	if (thread == THREAD_NULL) {
		return KERN_INVALID_ARGUMENT;
	}

	thread_mtx_lock(thread);

	if (invalid_policy(policy) ||
	    ((POLICY_TIMESHARE | POLICY_RR | POLICY_FIFO) & policy) == 0) {
		thread_mtx_unlock(thread);

		return KERN_INVALID_POLICY;
	}

	if (set_limit) {
		/*
		 *      Set scheduling limits to base priority.
		 */
		switch (policy) {
		case POLICY_RR:
		{
			policy_rr_base_t rr_base;

			if (count != POLICY_RR_BASE_COUNT) {
				result = KERN_INVALID_ARGUMENT;
				break;
			}

			limcount = POLICY_RR_LIMIT_COUNT;
			rr_base = (policy_rr_base_t) base;
			rr_limit.max_priority = rr_base->base_priority;
			limit = (policy_limit_t) &rr_limit;

			break;
		}

		case POLICY_FIFO:
		{
			policy_fifo_base_t fifo_base;

			if (count != POLICY_FIFO_BASE_COUNT) {
				result = KERN_INVALID_ARGUMENT;
				break;
			}

			limcount = POLICY_FIFO_LIMIT_COUNT;
			fifo_base = (policy_fifo_base_t) base;
			fifo_limit.max_priority = fifo_base->base_priority;
			limit = (policy_limit_t) &fifo_limit;

			break;
		}

		case POLICY_TIMESHARE:
		{
			policy_timeshare_base_t ts_base;

			if (count != POLICY_TIMESHARE_BASE_COUNT) {
				result = KERN_INVALID_ARGUMENT;
				break;
			}

			limcount = POLICY_TIMESHARE_LIMIT_COUNT;
			ts_base = (policy_timeshare_base_t) base;
			ts_limit.max_priority = ts_base->base_priority;
			limit = (policy_limit_t) &ts_limit;

			break;
		}

		default:
			result = KERN_INVALID_POLICY;
			break;
		}
	} else {
		/*
		 *	Use current scheduling limits. Ensure that the
		 *	new base priority will not exceed current limits.
		 */
		switch (policy) {
		case POLICY_RR:
		{
			policy_rr_base_t rr_base;

			if (count != POLICY_RR_BASE_COUNT) {
				result = KERN_INVALID_ARGUMENT;
				break;
			}

			limcount = POLICY_RR_LIMIT_COUNT;
			rr_base = (policy_rr_base_t) base;
			if (rr_base->base_priority > thread->max_priority) {
				result = KERN_POLICY_LIMIT;
				break;
			}

			rr_limit.max_priority = thread->max_priority;
			limit = (policy_limit_t) &rr_limit;

			break;
		}

		case POLICY_FIFO:
		{
			policy_fifo_base_t fifo_base;

			if (count != POLICY_FIFO_BASE_COUNT) {
				result = KERN_INVALID_ARGUMENT;
				break;
			}

			limcount = POLICY_FIFO_LIMIT_COUNT;
			fifo_base = (policy_fifo_base_t) base;
			if (fifo_base->base_priority > thread->max_priority) {
				result = KERN_POLICY_LIMIT;
				break;
			}

			fifo_limit.max_priority = thread->max_priority;
			limit = (policy_limit_t) &fifo_limit;

			break;
		}

		case POLICY_TIMESHARE:
		{
			policy_timeshare_base_t ts_base;

			if (count != POLICY_TIMESHARE_BASE_COUNT) {
				result = KERN_INVALID_ARGUMENT;
				break;
			}

			limcount = POLICY_TIMESHARE_LIMIT_COUNT;
			ts_base = (policy_timeshare_base_t) base;
			if (ts_base->base_priority > thread->max_priority) {
				result = KERN_POLICY_LIMIT;
				break;
			}

			ts_limit.max_priority = thread->max_priority;
			limit = (policy_limit_t) &ts_limit;

			break;
		}

		default:
			result = KERN_INVALID_POLICY;
			break;
		}
	}

	thread_mtx_unlock(thread);

	if (result == KERN_SUCCESS) {
		result = thread_set_policy(thread, pset,
		    policy, base, count, limit, limcount);
	}

	return result;
}
