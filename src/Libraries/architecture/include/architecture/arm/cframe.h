/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
/* Copyright (c) 1991 NeXT Software, Inc.  All rights reserved.
 *
 *	File:	architecture/ppc/cframe.h
 *	Author:	Mike DeMoney, NeXT Software, Inc.
 *
 *	This include file defines C calling sequence defines
 *	for ppc port.
 *
 * HISTORY
 * 20-May-97  Umesh Vaishampayan  (umeshv@apple.com)
 *	Added C_RED_ZONE.
 * 29-Dec-96  Umesh Vaishampayan  (umeshv@NeXT.com)
 *	Ported from m98k.
 * 11-June-91  Mike DeMoney (mike@next.com)
 *	Created.
 */

#ifndef	_ARCH_ARM_CFRAME_H_
#define	_ARCH_ARM_CFRAME_H_

/* Note that these values are copies of the somewhat more authoritative
 * values in <architecture/ppc/mode_independent_asm.h>.  We do not
 * include that file to avoid breaking legacy clients due to name
 * collisions.
 *
 * Note also that C_ARGSAVE_LEN isn't well defined or useful in PPC.
 * Most legacy uses of it are assuming it is the minimum stack frame
 * size, which is what we define it to be.
 */
#if defined(__arm__)
#define	C_ARGSAVE_LEN	32      /* "minimum arg save area" (but see above) */
#define	C_STACK_ALIGN	16      /* stack must be 16 byte aligned */
#define	C_RED_ZONE      224     /* 224 bytes to skip over saved registers */
#elif defined (__arm64__)
#define	C_STACK_ALIGN	16      /* stack must be 32 byte aligned */
#else /* !defined(__arm__) && !defined(__arm64__) */
#error Unknown architecture
#endif /* !defined(__arm__) && !defined(__arm64__) */

#endif	/* _ARCH_ARM_CFRAME_H_ */
