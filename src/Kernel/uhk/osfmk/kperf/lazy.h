/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#ifndef KPERF_LAZY_H
#define KPERF_LAZY_H

#include <stdbool.h>
#include <stdint.h>
#include <kern/thread.h>

extern unsigned int kperf_lazy_wait_action;
extern unsigned int kperf_lazy_cpu_action;

void kperf_lazy_reset(void);
void kperf_lazy_off_cpu(thread_t thread);
void kperf_lazy_make_runnable(thread_t thread, bool in_interrupt);
void kperf_lazy_wait_sample(thread_t thread,
    thread_continue_t continuation, uintptr_t *starting_fp);
void kperf_lazy_cpu_sample(thread_t thread, unsigned int flags, bool interrupt);

/* accessors for configuration */
int kperf_lazy_get_wait_action(void);
int kperf_lazy_get_cpu_action(void);
int kperf_lazy_set_wait_action(int action_id);
int kperf_lazy_set_cpu_action(int action_id);
uint64_t kperf_lazy_get_wait_time_threshold(void);
uint64_t kperf_lazy_get_cpu_time_threshold(void);
int kperf_lazy_set_wait_time_threshold(uint64_t threshold);
int kperf_lazy_set_cpu_time_threshold(uint64_t threshold);

#endif /* !defined(KPERF_LAZY_H) */
