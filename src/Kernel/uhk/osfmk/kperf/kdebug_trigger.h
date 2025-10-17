/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
#ifndef KPERF_KDEBUG_TRIGGER_H
#define KPERF_KDEBUG_TRIGGER_H

#include <mach/boolean.h>
#include <stdint.h>

#define KPERF_KDEBUG_DEBUGIDS_MAX (32)

struct kperf_kdebug_filter;

#define KPERF_KDEBUG_FILTER_SIZE(N_DEBUGIDS) ((2 * sizeof(uint64_t)) + ((N_DEBUGIDS) * sizeof(uint32_t)))
/* UNSAFE */
#define KPERF_KDEBUG_N_DEBUGIDS(FILTER_SIZE) \
	(((FILTER_SIZE) <= (2 * sizeof(uint64_t))) ? 0 : \
	  (((FILTER_SIZE) - (2 * sizeof(uint64_t))) / sizeof(uint32_t)))

void kperf_kdebug_setup(void);
void kperf_kdebug_reset(void);

boolean_t kperf_kdebug_should_trigger(uint32_t debugid);

int kperf_kdebug_set_action(int action_id);
int kperf_kdebug_get_action(void);

int kperf_kdebug_set_n_debugids(uint32_t n_debugids_in);
int kperf_kdebug_set_filter(user_addr_t user_filter, uint32_t user_size);
uint32_t kperf_kdebug_get_filter(struct kperf_kdebug_filter **filter);

#endif /* !defined(KPERF_KDEBUG_TRIGGER_H) */
