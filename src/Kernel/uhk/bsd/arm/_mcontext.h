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
#ifndef __ARM_MCONTEXT_H_
#define __ARM_MCONTEXT_H_

#if defined (__arm__) || defined (__arm64__)

#include <sys/cdefs.h> /* __DARWIN_UNIX03 */
#include <sys/appleapiopts.h>
#include <mach/machine/_structs.h>

#ifndef _STRUCT_MCONTEXT32
#if __DARWIN_UNIX03
#define _STRUCT_MCONTEXT32        struct __darwin_mcontext32
_STRUCT_MCONTEXT32
{
	_STRUCT_ARM_EXCEPTION_STATE     __es;
	_STRUCT_ARM_THREAD_STATE        __ss;
	_STRUCT_ARM_VFP_STATE           __fs;
};

#else /* !__DARWIN_UNIX03 */
#define _STRUCT_MCONTEXT32        struct mcontext32
_STRUCT_MCONTEXT32
{
	_STRUCT_ARM_EXCEPTION_STATE     es;
	_STRUCT_ARM_THREAD_STATE        ss;
	_STRUCT_ARM_VFP_STATE           fs;
};

#endif /* __DARWIN_UNIX03 */
#endif /* _STRUCT_MCONTEXT32 */


#ifndef _STRUCT_MCONTEXT64
#if __DARWIN_UNIX03
#define _STRUCT_MCONTEXT64      struct __darwin_mcontext64
_STRUCT_MCONTEXT64
{
	_STRUCT_ARM_EXCEPTION_STATE64   __es;
	_STRUCT_ARM_THREAD_STATE64      __ss;
	_STRUCT_ARM_NEON_STATE64        __ns;
};

#else /* !__DARWIN_UNIX03 */
#define _STRUCT_MCONTEXT64      struct mcontext64
_STRUCT_MCONTEXT64
{
	_STRUCT_ARM_EXCEPTION_STATE64   es;
	_STRUCT_ARM_THREAD_STATE64      ss;
	_STRUCT_ARM_NEON_STATE64        ns;
};
#endif /* __DARWIN_UNIX03 */
#endif /* _STRUCT_MCONTEXT32 */

#ifndef _MCONTEXT_T
#define _MCONTEXT_T
#if defined(__arm64__)
typedef _STRUCT_MCONTEXT64      *mcontext_t;
#define _STRUCT_MCONTEXT _STRUCT_MCONTEXT64
#else
typedef _STRUCT_MCONTEXT32      *mcontext_t;
#define _STRUCT_MCONTEXT        _STRUCT_MCONTEXT32
#endif
#endif /* _MCONTEXT_T */

#endif /* defined (__arm__) || defined (__arm64__) */

#endif /* __ARM_MCONTEXT_H_ */
