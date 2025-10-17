/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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
#ifndef _I386_VMX_ASM_H_
#define _I386_VMX_ASM_H_

#define VMX_FAIL_INVALID	-1
#define VMX_FAIL_VALID		-2
#define VMX_SUCCEED			0

/*
 * VMX Capability Registers (VCR)
 *
 */
#define VMX_VCR_VMCS_MEM_TYPE_BIT	50
#define VMX_VCR_VMCS_MEM_TYPE_MASK	0xF

#define VMX_VCR_VMCS_SIZE_BIT		32
#define VMX_VCR_VMCS_SIZE_MASK		0x01FFF
#define VMX_VCR_VMCS_REV_ID		0x00000000FFFFFFFFLL

#define VMX_VCR_ACT_HLT_BIT		6
#define VMX_VCR_ACT_HLT_MASK		0x1
#define VMX_VCR_ACT_SHUTDOWN_BIT	7
#define VMX_VCR_ACT_SHUTDOWN_MASK	0x1
#define VMX_VCR_ACT_SIPI_BIT		8
#define VMX_VCR_ACT_SIPI_MASK		0x1
#define VMX_VCR_ACT_CSTATE_BIT		9
#define VMX_VCR_ACT_CSTATE_MASK		0x1
#define VMX_VCR_CR3_TARGS_BIT		16
#define VMX_VCR_CR3_TARGS_MASK		0xFF
#define VMX_VCR_MAX_MSRS_BIT		25
#define VMX_VCR_MAX_MSRS_MASK		0x7
#define VMX_VCR_MSEG_ID_BIT		32
#define VMX_VCR_MSEG_ID_MASK		0xFFFFFFFF

#endif	/* _I386_VMX_H_ */
