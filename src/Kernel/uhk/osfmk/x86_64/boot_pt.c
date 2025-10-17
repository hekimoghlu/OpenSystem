/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#include <i386/pmap.h>

/*
 * These pagetables are used during early processor startup during
 * the transition from protected mode to 64-bit mode and the jump
 * to high kernel address space.
 *
 * They are required to be at the base of the kernel and specifically
 * the base of the special __HIB section.
 *
 * These tables are statically-defined as physical-zero-based.
 * Startup code in start.s rebases these according to the actual physical
 * base address.
 */

/*
 * NB: This must be located at the kernel's base address!
 */
#define PML4_PROT (INTEL_PTE_VALID | INTEL_PTE_WRITE)
pml4_entry_t    BootPML4[PTE_PER_PAGE]
__attribute__((section("__HIB, __bootPT"))) = {
	[0]                     = ((uint64_t)(PAGE_SIZE) | PML4_PROT),
	[KERNEL_PML4_INDEX]     = ((uint64_t)(PAGE_SIZE) | PML4_PROT),
};

#define PDPT_PROT (INTEL_PTE_VALID | INTEL_PTE_WRITE)
pdpt_entry_t    BootPDPT[PTE_PER_PAGE]
__attribute__((section("__HIB, __bootPT"))) = {
	[0]     = ((uint64_t)(2 * PAGE_SIZE) | PDPT_PROT),
	[1]     = ((uint64_t)(3 * PAGE_SIZE) | PDPT_PROT),
	[2]     = ((uint64_t)(4 * PAGE_SIZE) | PDPT_PROT),
	[3]     = ((uint64_t)(5 * PAGE_SIZE) | PDPT_PROT),
};

#if NPGPTD != 4
#error Please update boot_pt.c to reflect the new value of NPGPTD
#endif

#if MACHINE_BOOTSTRAPPTD

#define PDT_PROT (INTEL_PTE_PS | INTEL_PTE_VALID | INTEL_PTE_WRITE)
#define ID_MAP_2MEG(x)  [(x)] = ((((uint64_t)(x)) << 21) | (PDT_PROT)),

#define L0(x, n)  x(n)
#define L1(x, n)  L0(x,n-1)     L0(x,n)
#define L2(x, n)  L1(x,n-2)     L1(x,n)
#define L3(x, n)  L2(x,n-4)     L2(x,n)
#define L4(x, n)  L3(x,n-8)     L3(x,n)
#define L5(x, n)  L4(x,n-16)    L4(x,n)
#define L6(x, n)  L5(x,n-32)    L5(x,n)
#define L7(x, n)  L6(x,n-64)    L6(x,n)
#define L8(x, n)  L7(x,n-128)   L7(x,n)
#define L9(x, n)  L8(x,n-256)   L8(x,n)
#define L10(x, n) L9(x,n-512)   L9(x,n)
#define L11(x, n) L10(x,n-1024) L10(x,n)

#define FOR_0_TO_2047(x) L11(x,2047)

pd_entry_t      BootPTD[2048]
__attribute__((section("__HIB, __bootPT"))) = {
	FOR_0_TO_2047(ID_MAP_2MEG)
};
#endif /* MACHINE_BOOTSTRAPPTD */
