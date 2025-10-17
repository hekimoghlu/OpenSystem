/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#include <libsa/string.h>
#include <mach/vm_types.h>
#include <machine/static_if.h>

void
ml_static_if_entry_patch(static_if_entry_t sie, int branch)
{
	uint8_t insn[STATIC_IF_INSN_SIZE] = { 0x0F, 0x1F, 0x44, 0x00 };
	vm_offset_t patch_point = __static_if_entry_patch_point(sie);

	if (branch) {
		int32_t delta = (int32_t)(sie->sie_target -
		    (sie->sie_base + STATIC_IF_INSN_SIZE));
		insn[0] = 0xE9; /* jmp 32 */
		memcpy(insn + 1, &delta, sizeof(delta));
	}

	bcopy(insn, (void *)patch_point, STATIC_IF_INSN_SIZE);
}

void
ml_static_if_flush_icache(void)
{
}
