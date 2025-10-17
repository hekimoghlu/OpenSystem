/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#ifndef _ARM64_MACHINE_CPUID_H_
#define _ARM64_MACHINE_CPUID_H_

typedef struct {
	uint64_t        el0_not_implemented             : 1,
	    el0_aarch64_only                : 1,
	    el0_aarch32_and_64              : 1,
	    el1_not_implemented             : 1,
	    el1_aarch64_only                : 1,
	    el1_aarch32_and_64              : 1,
	    el2_not_implemented             : 1,
	    el2_aarch64_only                : 1,
	    el2_aarch32_and_64              : 1,
	    el3_not_implemented             : 1,
	    el3_aarch64_only                : 1,
	    el3_aarch32_and_64              : 1,
	    reserved                                : 52;
} arm_feature_bits_t;

/* Debug identification */

/* ID_AA64DFR0_EL1 */
typedef union {
	struct {
		uint64_t debug_arch_version             : 4,
		    trace_extn_version             : 4,
		    perf_extn_version              : 4,
		    brps                                   : 4,
		    reserved0                              : 4,
		    wrps                                   : 4,
		    reserved1                              : 4,
		    ctx_cmps                               : 4,
		    reserved32                             : 32;
	} debug_feature;
	uint64_t value;
} arm_cpuid_id_aa64dfr0_el1;

typedef struct {
	uint32_t        num_watchpoint_pairs;
	uint32_t        num_breakpoint_pairs;
} arm_debug_info_t;

#endif /* _MACHINE_CPUID_H_ */
