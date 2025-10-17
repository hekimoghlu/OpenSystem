/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#ifndef KPERF_CONTEXT_H
#define KPERF_CONTEXT_H

#include <kern/thread.h>

/* context of what we're looking at */
struct kperf_context {
	/* who was running during the event */
	int cur_pid;
	thread_t cur_thread;
	task_t cur_task;
	uintptr_t *starting_fp;

	/* who caused the event */
	unsigned int trigger_type;
	unsigned int trigger_id;
};

#endif /* !defined(KPERF_CONTEXT_H) */
