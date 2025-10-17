/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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
#include  <sys/fasttrap_isa.h>

uint8_t dtrace_decode_arm64(uint32_t instr);

struct arm64_decode_entry {
	uint32_t mask;
	uint32_t value;
	uint8_t type;
};

struct arm64_decode_entry arm64_decode_table[] = {
	{ .mask = 0xFFFFFFFF, .value = FASTTRAP_ARM64_OP_VALUE_FUNC_ENTRY, .type = FASTTRAP_T_ARM64_STANDARD_FUNCTION_ENTRY },
	{ .mask = FASTTRAP_ARM64_OP_MASK_LDR_S_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_LDR_S_PC_REL, .type = FASTTRAP_T_ARM64_LDR_S_PC_REL },
	{ .mask = FASTTRAP_ARM64_OP_MASK_LDR_W_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_LDR_W_PC_REL, .type = FASTTRAP_T_ARM64_LDR_W_PC_REL },
	{ .mask = FASTTRAP_ARM64_OP_MASK_LDR_D_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_LDR_D_PC_REL, .type = FASTTRAP_T_ARM64_LDR_D_PC_REL },
	{ .mask = FASTTRAP_ARM64_OP_MASK_LDR_X_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_LDR_X_PC_REL, .type = FASTTRAP_T_ARM64_LDR_X_PC_REL },
	{ .mask = FASTTRAP_ARM64_OP_MASK_LDR_Q_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_LDR_Q_PC_REL, .type = FASTTRAP_T_ARM64_LDR_Q_PC_REL },
	{ .mask = FASTTRAP_ARM64_OP_MASK_LRDSW_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_LRDSW_PC_REL, .type = FASTTRAP_T_ARM64_LDRSW_PC_REL },
	{ .mask = FASTTRAP_ARM64_OP_MASK_B_COND_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_B_COND_PC_REL, .type = FASTTRAP_T_ARM64_B_COND },
	{ .mask = FASTTRAP_ARM64_OP_MASK_CBNZ_W_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_CBNZ_W_PC_REL, .type = FASTTRAP_T_ARM64_CBNZ_W },
	{ .mask = FASTTRAP_ARM64_OP_MASK_CBNZ_X_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_CBNZ_X_PC_REL, .type = FASTTRAP_T_ARM64_CBNZ_X },
	{ .mask = FASTTRAP_ARM64_OP_MASK_CBZ_W_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_CBZ_W_PC_REL, .type = FASTTRAP_T_ARM64_CBZ_W },
	{ .mask = FASTTRAP_ARM64_OP_MASK_CBZ_X_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_CBZ_X_PC_REL, .type = FASTTRAP_T_ARM64_CBZ_X },
	{ .mask = FASTTRAP_ARM64_OP_MASK_TBNZ_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_TBNZ_PC_REL, .type = FASTTRAP_T_ARM64_TBNZ },
	{ .mask = FASTTRAP_ARM64_OP_MASK_TBZ_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_TBZ_PC_REL, .type = FASTTRAP_T_ARM64_TBZ },
	{ .mask = FASTTRAP_ARM64_OP_MASK_B_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_B_PC_REL, .type = FASTTRAP_T_ARM64_B },
	{ .mask = FASTTRAP_ARM64_OP_MASK_BL_PC_REL, .value = FASTTRAP_ARM64_OP_VALUE_BL_PC_REL, .type = FASTTRAP_T_ARM64_BL },
	{ .mask = FASTTRAP_ARM64_OP_MASK_BLR, .value = FASTTRAP_ARM64_OP_VALUE_BLR, .type = FASTTRAP_T_ARM64_BLR },
	{ .mask = FASTTRAP_ARM64_OP_MASK_BR, .value = FASTTRAP_ARM64_OP_VALUE_BR, .type = FASTTRAP_T_ARM64_BR },
	{ .mask = FASTTRAP_ARM64_OP_MASK_RET, .value = FASTTRAP_ARM64_OP_VALUE_RET, .type = FASTTRAP_T_ARM64_RET },
	{ .mask = FASTTRAP_ARM64_OP_MASK_ADRP, .value = FASTTRAP_ARM64_OP_VALUE_ADRP, .type = FASTTRAP_T_ARM64_ADRP },
	{ .mask = FASTTRAP_ARM64_OP_MASK_ADR, .value = FASTTRAP_ARM64_OP_VALUE_ADR, .type = FASTTRAP_T_ARM64_ADR },
	{ .mask = FASTTRAP_ARM64_OP_MASK_PRFM, .value = FASTTRAP_ARM64_OP_VALUE_PRFM, .type = FASTTRAP_T_ARM64_PRFM },
	{ .mask = FASTTRAP_ARM64_OP_MASK_EXCL_MEM, .value = FASTTRAP_ARM64_OP_VALUE_EXCL_MEM, .type = FASTTRAP_T_ARM64_EXCLUSIVE_MEM },
	{ .mask = FASTTRAP_ARM64_OP_MASK_RETAB, .value = FASTTRAP_ARM64_OP_VALUE_RETAB, .type = FASTTRAP_T_ARM64_RETAB }
};

#define NUM_DECODE_ENTRIES (sizeof(arm64_decode_table) / sizeof(struct arm64_decode_entry))

uint8_t
dtrace_decode_arm64(uint32_t instr)
{
	unsigned i;

	for (i = 0; i < NUM_DECODE_ENTRIES; i++) {
		if ((instr & arm64_decode_table[i].mask) == arm64_decode_table[i].value) {
			return arm64_decode_table[i].type;
		}
	}

	return FASTTRAP_T_COMMON;
}
