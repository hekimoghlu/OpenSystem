/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
#include <mach/vm_types.h>
#include <machine/static_if.h>
#include <arm64/amcc_rorgn.h>
#include <arm64/proc_reg.h>
#include <kern/startup.h>

extern char __text_exec_start[] __SEGMENT_START_SYM("__TEXT_EXEC");
extern char __text_exec_end[]   __SEGMENT_END_SYM("__TEXT_EXEC");

__attribute__((always_inline))
static uint32_t
arm64_insn_nop(void)
{
	return 0xd503201f;
}

__attribute__((always_inline))
static uint32_t
arm64_insn_b(int32_t delta)
{
	return 0x14000000u | ((delta >> 2) & 0x03ffffff);
}

MARK_AS_FIXUP_TEXT void
ml_static_if_entry_patch(static_if_entry_t sie, int branch)
{
	vm_offset_t patch_point = __static_if_entry_patch_point(sie);
	uint32_t insn;

	if (branch) {
		insn = arm64_insn_b(sie->sie_target);
	} else {
		insn = arm64_insn_nop();
	}

	if ((vm_offset_t)__text_exec_start <= patch_point &&
	    patch_point < (vm_offset_t)__text_exec_end) {
		asm volatile (""
		     /* patch the instruction */
                     "str     %w1, [%0]"     "\n\t"
#if !__ARM_IC_NOALIAS_ICACHE__
		     /* invalidate icache cacheline */
                     "ic      ivau, %0"      "\n\t"
                     "dsb     sy"            "\n\t"
                     "isb     sy"
#endif /* !__ARM_IC_NOALIAS_ICACHE__ */
                     : : "r"(patch_point), "r"(insn) : "memory");
	}
}

MARK_AS_FIXUP_TEXT void
ml_static_if_flush_icache(void)
{
	asm volatile (""
             "ic      ialluis"       "\n\t"
             "dsb     sy"            "\n\t"
             "isb     sy" : : : "memory");
}
