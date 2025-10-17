/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
 * This is the reference code for platform-specific implementations
 * of combined copy and 16-bit one's complement sum.
 */

#if !defined(__arm__) && !defined(__arm64__) && !defined(__x86_64__)
#include <sys/param.h>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <sys/conf.h>
#ifndef KERNEL
#include <strings.h>
#ifndef LIBSYSCALL_INTERFACE
#error "LIBSYSCALL_INTERFACE not defined"
#endif /* !LIBSYSCALL_INTERFACE */
#endif /* !KERNEL */

extern uint32_t os_cpu_copy_in_cksum(void *__sized_by(len), void *__sized_by(len),
    uint32_t len, uint32_t);
extern uint32_t os_cpu_in_cksum(const void *, uint32_t, uint32_t);

uint32_t
os_cpu_copy_in_cksum(void *__sized_by(len) src, void *__sized_by(len) dst,
    uint32_t len, uint32_t sum0)
{
	bcopy(src, dst, len);
	return os_cpu_in_cksum(dst, len, sum0);
}
#endif /* !__arm__ && !__arm64__ && !__x86_64__ */
