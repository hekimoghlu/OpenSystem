/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
#ifndef KPERF_THREAD_SAMPLERS_H
#define KPERF_THREAD_SAMPLERS_H

#include <kperf/context.h>

/* legacy thread info */
struct kperf_thread_info {
	uint64_t kpthi_pid;
	uint64_t kpthi_tid;
	uint64_t kpthi_dq_addr;
	uint64_t kpthi_runmode;
};

void kperf_thread_info_sample(struct kperf_thread_info *,
    struct kperf_context *);
void kperf_thread_info_log(struct kperf_thread_info *);

// legacy names
#define kpthsc_requested_qos_ipc_override kpthsc_requested_qos_kevent_override

/* scheduling information */
struct kperf_thread_scheduling {
	uint64_t kpthsc_user_time;
	uint64_t kpthsc_system_time;
	uint64_t kpthsc_runnable_time;
	unsigned int kpthsc_state;
	uint16_t kpthsc_base_priority;
	uint16_t kpthsc_sched_priority;
	unsigned int kpthsc_effective_qos :3,
	    kpthsc_requested_qos :3,
	    kpthsc_requested_qos_override :3,
	    kpthsc_requested_qos_promote :3,
	    kpthsc_requested_qos_kevent_override :3,
	    kpthsc_requested_qos_sync_ipc_override :3,             /* obsolete */
	    kpthsc_effective_latency_qos :3;
};

void kperf_thread_scheduling_sample(struct kperf_thread_scheduling *,
    struct kperf_context *);
void kperf_thread_scheduling_log(struct kperf_thread_scheduling *);

/* thread snapshot information */
struct kperf_thread_snapshot {
	uint64_t kpthsn_last_made_runnable_time;
	int16_t kpthsn_suspend_count;
	uint8_t kpthsn_io_tier;
	uint8_t kpthsn_flags;
};

void kperf_thread_snapshot_sample(struct kperf_thread_snapshot *,
    struct kperf_context *);
void kperf_thread_snapshot_log(struct kperf_thread_snapshot *);

/* libdispatch information */
struct kperf_thread_dispatch {
	uint64_t kpthdi_dq_serialno;
	char kpthdi_dq_label[64];
};

void kperf_thread_dispatch_sample(struct kperf_thread_dispatch *,
    struct kperf_context *);
int kperf_thread_dispatch_pend(struct kperf_context *, unsigned int actionid);
void kperf_thread_dispatch_log(struct kperf_thread_dispatch *);

void kperf_thread_inscyc_log(struct kperf_context *);

#endif /* !defined(KPERF_THREAD_SAMPLERS_H) */
