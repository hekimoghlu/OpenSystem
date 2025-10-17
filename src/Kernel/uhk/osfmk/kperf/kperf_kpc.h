/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
#ifndef __KPERF_KPC_H__
#define __KPERF_KPC_H__

#if CONFIG_CPU_COUNTERS

#include <kern/kpc.h> /* KPC_MAX_COUNTERS */

/* KPC sample data */
struct kpcdata {
	int      curcpu;
	uint32_t running;
	uint32_t counterc;
	uint64_t counterv[KPC_MAX_COUNTERS + 1];
	uint32_t configc;
	uint64_t configv[KPC_MAX_COUNTERS];
};

void kperf_kpc_thread_sample(struct kpcdata *, int);
void kperf_kpc_cpu_sample(struct kpcdata *, int);
void kperf_kpc_thread_log(const struct kpcdata *);
void kperf_kpc_cpu_log(const struct kpcdata *);
void kperf_kpc_config_log(const struct kpcdata *);

#endif /* CONFIG_CPU_COUNTERS */

#endif /* __KPERF_KPC_H__ */
