/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
#ifndef _MACH_ARM_PROCESSOR_INFO_H_
#define _MACH_ARM_PROCESSOR_INFO_H_

#if defined (__arm__) || defined (__arm64__)

#define PROCESSOR_CPU_STAT   0x10000003 /* Low-level CPU statistics */
#define PROCESSOR_CPU_STAT64 0x10000004 /* Low-level CPU statistics, in full 64-bit */

#include <stdint.h> /* uint32_t, uint64_t */

struct processor_cpu_stat {
	uint32_t irq_ex_cnt;
	uint32_t ipi_cnt;
	uint32_t timer_cnt;
	uint32_t undef_ex_cnt;
	uint32_t unaligned_cnt;
	uint32_t vfp_cnt;
	uint32_t vfp_shortv_cnt;
	uint32_t data_ex_cnt;
	uint32_t instr_ex_cnt;
};

typedef struct processor_cpu_stat  processor_cpu_stat_data_t;
typedef struct processor_cpu_stat *processor_cpu_stat_t;
#define PROCESSOR_CPU_STAT_COUNT ((mach_msg_type_number_t) \
	        (sizeof(processor_cpu_stat_data_t) / sizeof(natural_t)))

struct processor_cpu_stat64 {
	uint64_t irq_ex_cnt;
	uint64_t ipi_cnt;
	uint64_t timer_cnt;
	uint64_t undef_ex_cnt;
	uint64_t unaligned_cnt;
	uint64_t vfp_cnt;
	uint64_t vfp_shortv_cnt;
	uint64_t data_ex_cnt;
	uint64_t instr_ex_cnt;
	uint64_t pmi_cnt;
} __attribute__((packed, aligned(4)));

typedef struct processor_cpu_stat64  processor_cpu_stat64_data_t;
typedef struct processor_cpu_stat64 *processor_cpu_stat64_t;
#define PROCESSOR_CPU_STAT64_COUNT ((mach_msg_type_number_t) \
	        (sizeof(processor_cpu_stat64_data_t) / sizeof(integer_t)))

#endif /* defined (__arm__) || defined (__arm64__) */

#endif /* _MACH_ARM_PROCESSOR_INFO_H_ */
