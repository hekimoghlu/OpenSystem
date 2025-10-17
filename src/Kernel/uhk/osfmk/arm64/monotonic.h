/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#ifndef ARM64_MONOTONIC_H
#define ARM64_MONOTONIC_H

#include <stdbool.h>
#include <sys/cdefs.h>

#if CONFIG_CPU_COUNTERS

#include <pexpert/arm64/board_config.h>

#if KERNEL_PRIVATE

__BEGIN_DECLS

extern const bool mt_core_supported;

#if !CPMU_AIC_PMI
#define MONOTONIC_FIQ 1
#endif /* !CPMU_AIC_PMI */

#include <stdint.h>

#if HAS_UNCORE_CTRS
#define MT_NDEVS 2
#else /* HAS_UNCORE_CTRS */
#define MT_NDEVS 1
#endif /* !HAS_UNCORE_CTRS */

#define MT_CORE_CYCLES 0
#define MT_CORE_INSTRS 1
#define MT_CORE_NFIXED 2
#define MT_CORE_MAXVAL ((UINT64_C(1) << 48) - 1)

__END_DECLS

#endif /* KERNEL_PRIVATE */

#if MACH_KERNEL_PRIVATE

#include <stdbool.h>

__BEGIN_DECLS

/* set by hardware if a PMI was delivered */
#define PMCR0_PMAI (UINT64_C(1) << 11)
#define PMCR0_PMI(REG) ((REG) & PMCR0_PMAI)

#if HAS_UNCORE_CTRS

#define UPMSR_PMI(REG) ((REG) & 0x1)

#endif /* HAS_UNCORE_CTRS */

static inline bool
mt_pmi_pending(uint64_t * restrict pmcr0_out,
    uint64_t * restrict upmsr_out)
{
	uint64_t pmcr0 = __builtin_arm_rsr64("S3_1_C15_C0_0");
	bool pmi = PMCR0_PMI(pmcr0);
	if (pmi) {
		/*
		 * Acknowledge the PMI by clearing the pmai bit.
		 */
		__builtin_arm_wsr64("S3_1_C15_C0_0", pmcr0 & ~PMCR0_PMAI);
	}
	*pmcr0_out = pmcr0;

#if HAS_UNCORE_CTRS
	extern bool mt_uncore_enabled;
	if (mt_uncore_enabled) {
		uint64_t upmsr = __builtin_arm_rsr64("S3_7_C15_C6_4");
		if (UPMSR_PMI(upmsr)) {
			pmi = true;
		}
		*upmsr_out = upmsr;
	}
#else /* HAS_UNCORE_CTRS */
#pragma unused(upmsr_out)
#endif /* !HAS_UNCORE_CTRS */

	return pmi;
}

void mt_fiq(void *cpu, uint64_t pmcr0, uint64_t upmsr);

#if CPMU_AIC_PMI
void mt_cpmu_aic_pmi(void *source);
#endif /* CPMU_AIC_PMI */

__END_DECLS

#endif /* MACH_KERNEL_PRIVATE */

#endif /* CONFIG_CPU_COUNTERS */

#endif /* !defined(ARM64_MONOTONIC_H) */
