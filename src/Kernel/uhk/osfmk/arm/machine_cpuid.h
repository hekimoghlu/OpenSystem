/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
#ifndef _ARM_MACHINE_CPUID_H_
#define _ARM_MACHINE_CPUID_H_

/* CPU feature identification */

typedef struct {
	uint32_t        arm_32bit_isa   : 4,
	    arm_thumb_ver   : 4,
	    arm_jazelle             : 4,
	    arm_thumb2              : 4,
	    reserved                : 16;
} arm_feature_bits_t;

typedef union {
	arm_feature_bits_t      field;
	uint32_t                        value;
} arm_feature0_reg_t;

// Register 0, subtype 21: Instruction Set Features #1
typedef struct{
	uint32_t endianness_support     : 4;
	uint32_t exception_1_support    : 4;
	uint32_t exception_2_support    : 4;
	uint32_t sign_zero_ext_support  : 4;
	uint32_t if_then_support        : 4;
	uint32_t immediate_support      : 4;
	uint32_t interworking_support   : 4;
	uint32_t jazelle_support        : 4;
}
syscp_ID_instructions_feat_1_reg;

typedef union {
	uint32_t value;
	syscp_ID_instructions_feat_1_reg field;
} arm_isa_feat1_reg;

arm_isa_feat1_reg machine_read_isa_feat1(void);

/* Debug identification */

/* ID_DFR0 */
typedef union {
	struct {
		uint32_t    coprocessor_core_debug      : 4,
		    coprocessor_secure_debug    : 4,
		    memory_mapped_core_debug    : 4,
		    coprocessor_trace_debug     : 4,
		    memory_mapped_trace_debug   : 4,
		    microcontroller_debug       : 4;
	} debug_feature;
	uint32_t value;
} arm_cpuid_id_dfr0;

/* DBGDIDR */
typedef union {
	struct {
		uint32_t    revision                    : 4,
		    variant                     : 4,
		: 4,
		    se_imp                      : 1,
		    pcsr_imp                    : 1,
		    nsuhd_imp                   : 1,
		: 1,
		    version                     : 4,
		    ctx_cmps                    : 4,
		    brps                        : 4,
		    wrps                        : 4;
	} debug_id;
	uint32_t value;
} arm_debug_dbgdidr;

typedef struct {
	boolean_t               memory_mapped_core_debug;
	boolean_t               coprocessor_core_debug;
	uint32_t                num_watchpoint_pairs;
	uint32_t                num_breakpoint_pairs;
} arm_debug_info_t;

#endif /* _ARM_MACHINE_CPUID_H_ */
