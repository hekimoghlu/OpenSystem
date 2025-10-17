/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#ifndef _WORKQUEUE_TRACE_H_
#define _WORKQUEUE_TRACE_H_

// General workqueue tracepoints, mostly for debugging
#define WQ_TRACE_WORKQUEUE_SUBCLASS 1
// Workqueue request scheduling tracepoints
#define WQ_TRACE_REQUESTS_SUBCLASS 2
// Subclasses 3 - 6 in DBG_PTHREAD are used by libpthread

// Workqueue quantum tracepoints
#define WQ_TRACE_QUANTUM_SUBCLASS 7
// Generic pthread tracepoints
#define WQ_TRACE_BSDTHREAD_SUBCLASS 16

#define TRACE_wq_pthread_exit \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x01)
#define TRACE_wq_workqueue_exit \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x02)
#define TRACE_wq_runthread \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x03)
#define TRACE_wq_death_call \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x05)
#define TRACE_wq_thread_block \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x09)
#define TRACE_wq_thactive_update \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x0a)
#define TRACE_wq_add_timer \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x0b)
#define TRACE_wq_start_add_timer \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x0c)
#define TRACE_wq_override_dispatch \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x14)
#define TRACE_wq_override_reset \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x15)
#define TRACE_wq_thread_create_failed \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x1d)
#define TRACE_wq_thread_terminate \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x1e)
#define TRACE_wq_thread_create \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x1f)
#define TRACE_wq_select_threadreq \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x20)
#define TRACE_wq_creator_select \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x23)
#define TRACE_wq_creator_yield \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x24)
#define TRACE_wq_constrained_admission \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x25)
#define TRACE_wq_wqops_reqthreads \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x26)
#define TRACE_wq_cooperative_admission \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_WORKQUEUE_SUBCLASS, 0x27)

#define TRACE_wq_create \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_REQUESTS_SUBCLASS, 0x01)
#define TRACE_wq_destroy \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_REQUESTS_SUBCLASS, 0x02)
#define TRACE_wq_thread_logical_run \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_REQUESTS_SUBCLASS, 0x03)
#define TRACE_wq_thread_request_initiate \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_REQUESTS_SUBCLASS, 0x05)
#define TRACE_wq_thread_request_modify \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_REQUESTS_SUBCLASS, 0x06)
#define TRACE_wq_thread_request_fulfill \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_REQUESTS_SUBCLASS, 0x08)

#define TRACE_bsdthread_set_qos_self \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_BSDTHREAD_SUBCLASS, 0x1)

#define TRACE_wq_quantum_arm \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_QUANTUM_SUBCLASS, 0x01)
#define TRACE_wq_quantum_expired \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_QUANTUM_SUBCLASS, 0x02)
#define TRACE_wq_quantum_disarm \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_QUANTUM_SUBCLASS, 0x03)
#define TRACE_wq_quantum_expiry_reevaluate \
	        KDBG_CODE(DBG_PTHREAD, WQ_TRACE_QUANTUM_SUBCLASS, 0x04)

#define WQ_TRACE(x, a, b, c, d) \
	        ({ KERNEL_DEBUG_CONSTANT(x, a, b, c, d, 0); })
#define WQ_TRACE_WQ(x, wq, b, c, d) \
	        ({ KERNEL_DEBUG_CONSTANT(x, proc_getpid((wq)->wq_proc), b, c, d, 0); })

#if (KDEBUG_LEVEL >= KDEBUG_LEVEL_STANDARD)
#define __wq_trace_only
#else // (KDEBUG_LEVEL >= KDEBUG_LEVEL_STANDARD)
#define __wq_trace_only __unused
#endif // (KDEBUG_LEVEL >= KDEBUG_LEVEL_STANDARD)

#endif // _WORKQUEUE_TRACE_H_
