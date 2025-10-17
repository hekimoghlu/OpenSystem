/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <mach/mach_time.h>
#include <sys/time.h>

#include <darwintest.h>
#include <darwintest_perf.h>

T_GLOBAL_META(T_META_TAG_PERF);

T_DECL(gettimeofday_tl, "gettimeofday performance in tight loop", T_META_TAG_VM_NOT_ELIGIBLE) {
	{
		struct timeval time;
		dt_stat_time_t s = dt_stat_time_create("gettimeofday tight loop");
		T_STAT_MEASURE_LOOP(s){
			gettimeofday(&time, NULL);
		}
		dt_stat_finalize(s);
	}
}

extern int __gettimeofday(struct timeval *, struct timezone *);
T_DECL(__gettimeofday_tl, "__gettimeofday performance in tight loop", T_META_TAG_VM_NOT_ELIGIBLE) {
	{
		struct timeval time;

		dt_stat_time_t s = dt_stat_time_create("__gettimeofday tight loop");
		T_STAT_MEASURE_LOOP(s){
			__gettimeofday(&time, NULL);
		}
		dt_stat_finalize(s);
	}
}

T_DECL(gettimeofday_sl, "gettimeofday performance in loop with sleep", T_META_TAG_VM_NOT_ELIGIBLE) {
	{
		struct timeval time;
		dt_stat_time_t s = dt_stat_time_create("gettimeofday loop with sleep");
		while (!dt_stat_stable(s)) {
			T_STAT_MEASURE_BATCH(s){
				gettimeofday(&time, NULL);
			}
			sleep(1);
		}
		dt_stat_finalize(s);
	}
}
