/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
 *	Copyright (C) 1990,  NeXT, Inc.
 *
 *	File:	next/kern_machdep.c
 *	Author:	John Seamons
 *
 *	Machine-specific kernel routines.
 */

#include        <sys/types.h>
#include        <mach/machine.h>
#include        <kern/cpu_number.h>
#include        <machine/exec.h>
#include        <machine/machine_routines.h>

#if __x86_64__
extern int bootarg_no32exec;    /* bsd_init.c */
#endif

/**********************************************************************
* Routine:	grade_binary()
*
* Function:	Say OK to CPU types that we can actually execute on the given
*		system. 64-bit binaries have the highest preference, followed
*		by 32-bit binaries. 0 means unsupported.
**********************************************************************/
int
grade_binary(cpu_type_t exectype, cpu_subtype_t execsubtype, cpu_subtype_t execfeatures __unused, bool allow_simulator_binary __unused)
{
	cpu_subtype_t hostsubtype = cpu_subtype();

	switch (exectype) {
	case CPU_TYPE_X86_64:           /* native 64-bit */
		switch (hostsubtype) {
		case CPU_SUBTYPE_X86_64_H:      /* x86_64h can execute anything */
			switch (execsubtype) {
			case CPU_SUBTYPE_X86_64_H:
				return 3;
			case CPU_SUBTYPE_X86_64_ALL:
				return 2;
			}
			break;
		case CPU_SUBTYPE_X86_ARCH1:     /* generic systems can only execute ALL subtype */
			switch (execsubtype) {
			case CPU_SUBTYPE_X86_64_ALL:
				return 2;
			}
			break;
		}
		break;
	case CPU_TYPE_X86:              /* native */
#if __x86_64__
		if (bootarg_no32exec && !allow_simulator_binary) {
			return 0;
		}
#endif
		return 1;
	}

	return 0;
}
