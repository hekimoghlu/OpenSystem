/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#include <arm/cpuid.h>
#include <arm/cpuid_internal.h>
#include <machine/atomic.h>
#include <machine/machine_cpuid.h>
#include <arm/cpu_data_internal.h>

static arm_mvfp_info_t cpuid_mvfp_info;
static arm_debug_info_t cpuid_debug_info;

uint32_t
machine_read_midr(void)
{
	uint64_t midr;
	__asm__ volatile ("mrs	%0, MIDR_EL1"  : "=r" (midr));

	return (uint32_t)midr;
}

uint32_t
machine_read_clidr(void)
{
	uint64_t clidr;
	__asm__ volatile ("mrs	%0, CLIDR_EL1"  : "=r" (clidr));

	return (uint32_t)clidr;
}

uint32_t
machine_read_ccsidr(void)
{
	uint64_t ccsidr;
	__asm__ volatile ("mrs	%0, CCSIDR_EL1"  : "=r" (ccsidr));

	return (uint32_t)ccsidr;
}

void
machine_write_csselr(csselr_cache_level level, csselr_cache_type type)
{
	uint64_t csselr = (uint64_t)level | (uint64_t)type;
	__asm__ volatile ("msr	CSSELR_EL1, %0"  : : "r" (csselr));

	__builtin_arm_isb(ISB_SY);
}

void
machine_do_debugid(void)
{
	arm_cpuid_id_aa64dfr0_el1 id_dfr0;

	/* read ID_AA64DFR0_EL1 */
	__asm__ volatile ("mrs %0, ID_AA64DFR0_EL1" : "=r"(id_dfr0.value));

	if (id_dfr0.debug_feature.debug_arch_version) {
		cpuid_debug_info.num_watchpoint_pairs = id_dfr0.debug_feature.wrps + 1;
		cpuid_debug_info.num_breakpoint_pairs = id_dfr0.debug_feature.brps + 1;
	}
}

arm_debug_info_t *
machine_arm_debug_info(void)
{
	return &cpuid_debug_info;
}

void
machine_do_mvfpid()
{
	cpuid_mvfp_info.neon = 1;
	cpuid_mvfp_info.neon_hpfp = 1;
#if defined(__ARM_ARCH_8_2__)
	cpuid_mvfp_info.neon_fp16 = 1;
#endif /* defined(__ARM_ARCH_8_2__) */
}

arm_mvfp_info_t *
machine_arm_mvfp_info(void)
{
	return &cpuid_mvfp_info;
}
