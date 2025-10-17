/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*	Copyright (c) 1990, 1991 UNIX System Laboratories, Inc. */

/*	Copyright (c) 1984, 1986, 1987, 1988, 1989 AT&T		*/
/*	All Rights Reserved	*/

#ifndef	_FASTTRAP_REGSET_H
#define	_FASTTRAP_REGSET_H

/*
 * APPLE NOTE: This file was orginally uts/intel/sys/regset.h
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The names and offsets defined here should be specified by the
 * AMD64 ABI suppl.
 *
 * We make fsbase and gsbase part of the lwp context (since they're
 * the only way to access the full 64-bit address range via the segment
 * registers) and thus belong here too.  However we treat them as
 * read-only; if %fs or %gs are updated, the results of the descriptor
 * table lookup that those updates implicitly cause will be reflected
 * in the corresponding fsbase and/or gsbase values the next time the
 * context can be inspected.  However it is NOT possible to override
 * the fsbase/gsbase settings via this interface.
 *
 * Direct modification of the base registers (thus overriding the
 * descriptor table base address) can be achieved with _lwp_setprivate.
 */

#define	REG_GSBASE	27
#define	REG_FSBASE	26
#define	REG_DS		25
#define	REG_ES		24

#define	REG_GS		23
#define	REG_FS		22
#define	REG_SS		21
#define	REG_RSP		20
#define	REG_RFL		19
#define	REG_CS		18
#define	REG_RIP		17
#define	REG_ERR		16
#define	REG_TRAPNO	15
#define	REG_RAX		14
#define	REG_RCX		13
#define	REG_RDX		12
#define	REG_RBX		11
#define	REG_RBP		10
#define	REG_RSI		9
#define	REG_RDI		8
#define	REG_R8		7
#define	REG_R9		6
#define	REG_R10		5
#define	REG_R11		4
#define	REG_R12		3
#define	REG_R13		2
#define	REG_R14		1
#define	REG_R15		0

/*
 * The names and offsets defined here are specified by i386 ABI suppl.
 */

#define	SS		18	/* only stored on a privilege transition */
#define	UESP		17	/* only stored on a privilege transition */
#define	EFL		16
#define	CS		15
#define	EIP		14
#define	ERR		13
#define	TRAPNO		12
#define	EAX		11
#define	ECX		10
#define	EDX		9
#define	EBX		8
#define	ESP		7
#define	EBP		6
#define	ESI		5
#define	EDI		4
#define	DS		3
#define	ES		2
#define	FS		1
#define	GS		0

#define REG_PC  EIP
#define REG_FP  EBP
#define REG_SP  UESP
#define REG_PS  EFL
#define REG_R0  EAX
#define REG_R1  EDX

#ifdef	__cplusplus
}
#endif

#endif	/* _FASTTRAP_REGSET_H */
