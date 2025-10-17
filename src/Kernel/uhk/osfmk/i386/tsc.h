/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
/*
 * @OSF_COPYRIGHT@
 */
/*
 * @APPLE_FREE_COPYRIGHT@
 */
/*
 *	File:		tsc.h
 *	Purpose:	Contains the TSC initialization and conversion
 *			factors.
 */
#ifdef KERNEL_PRIVATE
#ifndef _I386_TSC_H_
#define _I386_TSC_H_

#define BASE_NHM_CLOCK_SOURCE   133333333ULL
#define BASE_ART_CLOCK_SOURCE           24000000ULL     /* 24MHz */
#define BASE_ART_CLOCK_SOURCE_SP        25000000ULL     /* 25MHz */
#define IA32_PERF_STS                   0x198
#define SLOW_TSC_THRESHOLD      1000067800      /* if slower, nonzero shift required in nanotime() algorithm */

#ifndef ASSEMBLER
extern uint64_t busFCvtt2n;
extern uint64_t busFCvtn2t;
extern uint64_t tscFreq;
extern uint64_t tscFCvtt2n;
extern uint64_t tscFCvtn2t;
extern uint64_t tscGranularity;
extern uint64_t bus2tsc;
extern uint64_t busFreq;
extern uint32_t flex_ratio;
extern uint32_t flex_ratio_min;
extern uint32_t flex_ratio_max;
extern uint64_t tsc_at_boot;

struct tscInfo {
	uint64_t        busFCvtt2n;
	uint64_t        busFCvtn2t;
	uint64_t        tscFreq;
	uint64_t        tscFCvtt2n;
	uint64_t        tscFCvtn2t;
	uint64_t        tscGranularity;
	uint64_t        bus2tsc;
	uint64_t        busFreq;
	uint32_t        flex_ratio;
	uint32_t        flex_ratio_min;
	uint32_t        flex_ratio_max;
};
typedef struct tscInfo tscInfo_t;

extern void tsc_get_info(tscInfo_t *info);

extern void tsc_init(void);

#if DEVELOPMENT || DEBUG
extern void cpu_data_tsc_sync_deltas_string(char *buf, uint32_t buflen,
    uint32_t start_cpu, uint32_t end_cpu);
#endif

#endif /* ASSEMBLER */
#endif /* _I386_TSC_H_ */
#endif /* KERNEL_PRIVATE */
