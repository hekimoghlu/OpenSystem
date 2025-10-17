/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#ifndef KPERF_TASK_SAMPLERS_H
#define KPERF_TASK_SAMPLERS_H

#include <kperf/context.h>
#include <kern/task.h>

struct kperf_task_snapshot {
	uint64_t kptksn_flags;
	uint64_t kptksn_user_time_in_terminated_threads;
	uint64_t kptksn_system_time_in_terminated_threads;
	int kptksn_suspend_count;
	int kptksn_pageins;
};

#define KPERF_TASK_FLAG_DARWIN_BG               (1U << 0)
#define KPERF_TASK_FLAG_FOREGROUND              (1U << 1)
#define KPERF_TASK_FLAG_BOOSTED                 (1U << 2)
#define KPERF_TASK_FLAG_DIRTY                   (1U << 3)
#define KPERF_TASK_FLAG_WQ_FLAGS_VALID          (1U << 4)
#define KPERF_TASK_FLAG_WQ_EXCEEDED_TOTAL       (1U << 5)
#define KPERF_TASK_FLAG_WQ_EXCEEDED_CONSTRAINED (1U << 6)
#define KPERF_TASK_FLAG_DIRTY_TRACKED           (1U << 7)
#define KPERF_TASK_ALLOW_IDLE_EXIT              (1U << 8)

void kperf_task_snapshot_sample(task_t task, struct kperf_task_snapshot *tksn);
void kperf_task_snapshot_log(struct kperf_task_snapshot *tksn);
void kperf_task_info_log(struct kperf_context *ctx);

#endif /* !defined(KPERF_TASK_SAMPLERS_H) */
