/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
#ifndef KPERF_SAMPLE_H
#define KPERF_SAMPLE_H

#include <kperf/thread_samplers.h>
#include <kperf/task_samplers.h>
#include "callstack.h"
#include "kperf_kpc.h"
#include "meminfo.h"

/*
 * Dispatch sampling may need to read from compressed, file-backed pages, which
 * incurs a steep stack usage penalty.
 */
struct kperf_usample_min {
	struct kperf_thread_dispatch th_dispatch;
};

/*
 * For data that must be sampled in a fault-able context.
 */
struct kperf_usample {
	struct kperf_usample_min *usample_min;
	struct kp_ucallstack ucallstack;
	struct kperf_thread_info th_info;
};

struct kperf_sample {
	struct kperf_thread_info       th_info;
	struct kperf_thread_scheduling th_scheduling;
	struct kperf_thread_snapshot   th_snapshot;

	struct kperf_task_snapshot tk_snapshot;

	struct kp_kcallstack kcallstack;
	struct meminfo     meminfo;

	struct kperf_usample usample;

#if CONFIG_CPU_COUNTERS
	struct kpcdata    kpcdata;
#endif /* CONFIG_CPU_COUNTERS */

#if DEVELOPMENT || DEBUG
	uint64_t sample_time;
#endif /* DEVELOPMENT || DEBUG */
};

/* cache of threads on each CPU during a timer fire */
extern uint64_t *kperf_tid_on_cpus;

#endif /* !defined(KPERF_SAMPLE_H) */
