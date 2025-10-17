/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#ifndef MACHINE_MONOTONIC_H
#define MACHINE_MONOTONIC_H

#if CONFIG_CPU_COUNTERS

#if defined(__x86_64__)
#include <x86_64/monotonic.h>
#elif defined(__arm64__)
#include <arm64/monotonic.h>
#else
#error unsupported architecture
#endif

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

struct mt_cpu {
	uint64_t mtc_snaps[MT_CORE_NFIXED];
	uint64_t mtc_counts[MT_CORE_NFIXED];
	uint64_t mtc_counts_last[MT_CORE_NFIXED];
	uint64_t mtc_npmis;
	/*
	 * Whether this CPU should be using PMCs.
	 */
	bool mtc_active;
};

struct mt_cpu *mt_cur_cpu(void);

uint64_t mt_count_pmis(void);
void mt_mtc_update_fixed_counts(struct mt_cpu *mtc, uint64_t *counts,
    uint64_t *counts_since);
uint64_t mt_mtc_update_count(struct mt_cpu *mtc, unsigned int ctr);
uint64_t mt_core_snap(unsigned int ctr);
void mt_core_set_snap(unsigned int ctr, uint64_t snap);
void mt_mtc_set_snap(struct mt_cpu *mtc, unsigned int ctr, uint64_t snap);

typedef void (*mt_pmi_fn)(bool user_mode, void *ctx);
extern bool mt_microstackshots;
extern unsigned int mt_microstackshot_ctr;
extern mt_pmi_fn mt_microstackshot_pmi_handler;
extern void *mt_microstackshot_ctx;
extern uint64_t mt_core_reset_values[MT_CORE_NFIXED];
int mt_microstackshot_start_arch(uint64_t period);

#endif /* CONFIG_CPU_COUNTERS */

#endif /* !defined(MACHINE_MONOTONIC_H) */
