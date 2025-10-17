/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
 * This file is in the public domain.
 */

#ifndef	_MACHINE_ELF_H_
#define	_MACHINE_ELF_H_

/*
 * CPU Feature Attributes
 *
 * These are defined in the PowerPC ELF ABI for the AT_HWCAP vector,
 * and are exported to userland via the elf_aux_info(3) function.
 */

#ifdef _KERNEL
# define __HAVE_CPU_HWCAP
# define __HAVE_CPU_HWCAP2
extern unsigned long hwcap;
extern unsigned long hwcap2;
#endif /* _KERNEL */

#define	PPC_FEATURE_32		0x80000000	/* Always true */
#define	PPC_FEATURE_64		0x40000000	/* Defined on a 64-bit CPU */
#define	PPC_FEATURE_601_INSTR	0x20000000
#define	PPC_FEATURE_HAS_ALTIVEC	0x10000000
#define	PPC_FEATURE_HAS_FPU	0x08000000
#define	PPC_FEATURE_HAS_MMU	0x04000000
#define	PPC_FEATURE_UNIFIED_CACHE 0x01000000
#define	PPC_FEATURE_HAS_SPE	0x00800000
#define	PPC_FEATURE_HAS_EFP_SINGLE	0x00400000
#define	PPC_FEATURE_HAS_EFP_DOUBLE	0x00200000
#define	PPC_FEATURE_NO_TB	0x00100000
#define	PPC_FEATURE_POWER4	0x00080000
#define	PPC_FEATURE_POWER5	0x00040000
#define	PPC_FEATURE_POWER5_PLUS	0x00020000
#define	PPC_FEATURE_CELL	0x00010000
#define	PPC_FEATURE_BOOKE	0x00008000
#define	PPC_FEATURE_SMT		0x00004000
#define	PPC_FEATURE_ICACHE_SNOOP	0x00002000
#define	PPC_FEATURE_ARCH_2_05	0x00001000
#define	PPC_FEATURE_HAS_DFP	0x00000400
#define	PPC_FEATURE_POWER6_EXT	0x00000200
#define	PPC_FEATURE_ARCH_2_06	0x00000100
#define	PPC_FEATURE_HAS_VSX	0x00000080
#define	PPC_FEATURE_TRUE_LE	0x00000002
#define	PPC_FEATURE_PPC_LE	0x00000001

#define	PPC_FEATURE2_ARCH_2_07	0x80000000
#define	PPC_FEATURE2_HTM	0x40000000
#define	PPC_FEATURE2_DSCR	0x20000000
#define	PPC_FEATURE2_EBB	0x10000000
#define	PPC_FEATURE2_ISEL	0x08000000
#define	PPC_FEATURE2_TAR	0x04000000
#define	PPC_FEATURE2_HAS_VEC_CRYPTO	0x02000000
#define	PPC_FEATURE2_HTM_NOSC	0x01000000
#define	PPC_FEATURE2_ARCH_3_00	0x00800000
#define	PPC_FEATURE2_HAS_IEEE128	0x00400000
#define	PPC_FEATURE2_DARN	0x00200000
#define	PPC_FEATURE2_SCV	0x00100000
#define	PPC_FEATURE2_HTM_NOSUSPEND	0x00080000
#define	PPC_FEATURE2_ARCH_3_1	0x00040000
#define	PPC_FEATURE2_MMA	0x00020000

#endif /* !_MACHINE_ELF_H_ */
