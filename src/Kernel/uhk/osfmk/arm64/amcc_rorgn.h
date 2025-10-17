/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
#ifndef _ARM64_AMCC_RORGN_H_
#define _ARM64_AMCC_RORGN_H_

#include <sys/cdefs.h>
#include <stdbool.h>
#include <libkern/section_keywords.h>

__BEGIN_DECLS

#if defined(KERNEL_INTEGRITY_KTRR) || defined(KERNEL_INTEGRITY_CTRR)

extern vm_offset_t ctrr_begin, ctrr_end;

#if CONFIG_CSR_FROM_DT
extern bool csr_unsafe_kernel_text;
#endif /* CONFIG_CSR_FROM_DT */

#if DEVELOPMENT || DEBUG || CONFIG_DTRACE || CONFIG_CSR_FROM_DT
extern bool rorgn_disable;
#else
#define rorgn_disable false
#endif /* DEVELOPMENT || DEBUG */

void rorgn_stash_range(void);
void rorgn_lockdown(void);
bool rorgn_contains(vm_offset_t addr, vm_size_t size, bool defval);
void rorgn_validate_core(void);

#if KERNEL_CTRR_VERSION >= 3
#define CTXR_XN_DISALLOW_ALL \
	/* Execute Masks for EL2&0 */ \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_EL2_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_EL0TGE1_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_GL2_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_GL0TGE1_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_MMUOFF_shift) | \
	/* Execute Masks for EL1&0 when Stage2 Translation is disabled */ \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_EL1_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_EL0TGE0_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_GL1_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_GL0TGE0_shift)

#define CTXR_XN_KERNEL \
	/* Execute Masks for EL2&0 */ \
    (CTXR3_XN_disallow_outside << CTXR3_x_CTL_EL2_XN_EL2_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_EL0TGE1_shift) | \
    (CTXR3_XN_disallow_outside << CTXR3_x_CTL_EL2_XN_GL2_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_GL0TGE1_shift) | \
    (CTXR3_XN_disallow_outside << CTXR3_x_CTL_EL2_XN_MMUOFF_shift) | \
	/* Execute Masks for EL1&0 when Stage2 Translation is disabled */ \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_EL1_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_EL0TGE0_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_GL1_shift) | \
    (CTXR3_XN_disallow_inside << CTXR3_x_CTL_EL2_XN_GL0TGE0_shift)
#endif /* KERNEL_CTRR_VERSION >= 3 */

#else

#if CONFIG_CSR_FROM_DT
#define csr_unsafe_kernel_text false
#endif /* CONFIG_CSR_FROM_DT */

#define rorgn_disable false
#endif /* defined(KERNEL_INTEGRITY_KTRR) || defined(KERNEL_INTEGRITY_CTRR) */

__END_DECLS

#endif /* _ARM64_AMCC_RORGN_H_ */
