/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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
 * @OSF_FREE_COPYRIGHT@
 */
/*
 * @APPLE_FREE_COPYRIGHT@
 */

/*
 *	Here are the Diagnostic interface interfaces
 *	Lovingly crafted by Bill Angell using traditional methods
 */
#ifdef  KERNEL_PRIVATE

#ifndef _DIAGNOSTICS_H_
#define _DIAGNOSTICS_H_

#if !(defined(__i386__) || defined(__x86_64__))
#error This file is not useful on non-Intel
#endif

int diagCall64(x86_saved_state_t *regs);

#define diagSCnum 0x00006000

#define dgAdjTB 0
#define dgLRA 1
#define dgpcpy 2
#define dgreset 3
#define dgtest 4
#define dgBMphys 5
#define dgUnMap 6
#define dgBootScreen 7
#define dgFlush 8
#define dgAlign 9
#define dgGzallocTest 10
#define dgmck 11
#define dg64 12
#define dgProbeRead 13
#define dgCPNull 14
#define dgPerfMon 15
#define dgMapPage 16
#define dgPowerStat 17
#define dgBind 18
#define dgAcntg 20
#define dgKlra 21
#define dgEnaPMC 22
#define dgWar 23
#define dgNapStat 24
#define dgRuptStat 25
#define dgPermCheck 26

typedef struct diagWork {                       /* Diagnostic work area */
	unsigned int dgLock;                    /* Lock if needed */
	unsigned int dgFlags;                   /* Flags */
#define enaExpTrace 0x00000001
#define enaUsrFCall 0x00000002
#define enaUsrPhyMp 0x00000004
#define enaDiagSCs  0x00000008
#define enaDiagDM  0x00000010
#define enaDiagEM  0x00000020
#define enaDiagTrap  0x00000040
#define enaNotifyEM  0x00000080

	unsigned int dgMisc0;
	unsigned int dgMisc1;
	unsigned int dgMisc2;
	unsigned int dgMisc3;
	unsigned int dgMisc4;
	unsigned int dgMisc5;
} diagWork;

extern diagWork dgWork;

#define FIXED_PMC (1 << 30)
#define FIXED_S3_2_C15_C0_0 (FIXED_PMC)
#define FIXED_S3_2_C15_C1_0 (FIXED_PMC | 1)
#define FIXED_S3_2_C15_C2_0 (FIXED_PMC | 2)
#define GS3_2_C15_C0_0 (0)
#define GS3_2_C15_C1_0 (1)
#define GS3_2_C15_C2_0 (2)
#define GS3_2_C15_C3_0 (3)

static inline uint64_t
read_pmc(uint32_t counter)
{
	uint32_t lo = 0, hi = 0;
	__asm__ volatile ("rdpmc" : "=a" (lo), "=d" (hi) : "c" (counter));
	return (((uint64_t)hi) << 32) | ((uint64_t)lo);
}
#endif /* _DIAGNOSTICS_H_ */

#endif /* KERNEL_PRIVATE */
