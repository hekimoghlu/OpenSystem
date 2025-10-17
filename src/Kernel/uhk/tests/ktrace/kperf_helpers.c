/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
// Copyright (c) 2018-2020 Apple Inc.  All rights reserved.

#include "kperf_helpers.h"

#include <darwintest.h>
#include <kperf/kperf.h>
#include <unistd.h>

void
configure_kperf_stacks_timer(pid_t pid, unsigned int period_ms, bool quiet)
{
	kperf_reset();

	(void)kperf_action_count_set(1);
	(void)kperf_timer_count_set(1);

	if (quiet) {
		T_QUIET;
	}
	T_ASSERT_POSIX_SUCCESS(kperf_action_samplers_set(1,
	    KPERF_SAMPLER_USTACK | KPERF_SAMPLER_KSTACK), NULL);

	if (pid != -1) {
		if (quiet) {
			T_QUIET;
		}
		T_ASSERT_POSIX_SUCCESS(kperf_action_filter_set_by_pid(1, pid), NULL);
	}

	if (quiet) {
		T_QUIET;
	}
	T_ASSERT_POSIX_SUCCESS(kperf_timer_action_set(0, 1), NULL);
	if (quiet) {
		T_QUIET;
	}
	T_ASSERT_POSIX_SUCCESS(kperf_timer_period_set(0,
	    kperf_ns_to_ticks(period_ms * NSEC_PER_MSEC)), NULL);
}
