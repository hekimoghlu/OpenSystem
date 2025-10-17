/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
#ifndef _I386_VMX_CPU_H_
#define _I386_VMX_CPU_H_

#include <mach/machine/vm_types.h>
#include <mach/boolean.h>
#include <i386/vmx/vmx_asm.h>

/*
 * Physical CPU's VMX specifications
 *
 */
typedef struct vmx_specs {
	boolean_t       initialized;    /* the specs have already been read */
	boolean_t       vmx_present;    /* VMX feature available and enabled */
	boolean_t       vmx_on;                 /* VMX is active */
	uint32_t        vmcs_id;                /* VMCS revision identifier */
	/*
	 * Fixed control register bits are specified by a pair of
	 * bitfields: 0-settings contain 0 bits corresponding to
	 * CR bits that may be 0; 1-settings contain 1 bits
	 * corresponding to CR bits that may be 1.
	 */
	uint32_t        cr0_fixed_0;    /* allowed 0-settings for CR0 */
	uint32_t        cr0_fixed_1;    /* allowed 1-settings for CR0 */

	uint32_t        cr4_fixed_0;    /* allowed 0-settings for CR4 */
	uint32_t        cr4_fixed_1;    /* allowed 1-settings for CR4 */
} vmx_specs_t;

typedef struct vmx_cpu {
	vmx_specs_t     specs;          /* this phys CPU's VMX specifications */
	void            *vmxon_region;  /* the logical address of the VMXON region page */
} vmx_cpu_t;

void vmx_cpu_init(void);
void vmx_resume(boolean_t is_wake_from_hibernate);
void vmx_suspend(void);

#define VMX_BASIC_TRUE_CTLS                                     (1ull << 55)
#define VMX_TRUE_PROCBASED_SECONDARY_CTLS       (1ull << 31)
#define VMX_PROCBASED_CTLS2_EPT                         (1ull << 1)
#define VMX_PROCBASED_CTLS2_UNRESTRICTED        (1ull << 7)

#define VMX_CAP(msr, shift, mask) (rdmsr64(msr) & ((mask) << (shift)))

boolean_t vmx_hv_support(void);

/*
 *	__vmxoff -- Leave VMX Operation
 *
 */
extern int __vmxoff(void);

/*
 *	__vmxon -- Enter VMX Operation
 *
 */
extern int __vmxon(addr64_t v);

#endif  /* _I386_VMX_CPU_H_ */
