/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#ifndef _LIBKERN_COPYIO_H_
#define _LIBKERN_COPYIO_H_

#include <kern/debug.h>

__BEGIN_DECLS

int     copyin(const user_addr_t uaddr, void *__sized_by(len) kaddr, size_t len) OS_WARN_RESULT;
int     copyout(const void *__sized_by(len) kaddr, user_addr_t udaddr, size_t len);

#if defined (_FORTIFY_SOURCE) && _FORTIFY_SOURCE == 0
/* FORTIFY_SOURCE disabled (it is assumed to be 1 if undefined) */

#ifdef XNU_KERNEL_PRIVATE
/* copyio wrappers that return mach error code */
#define mach_copyin(uaddr, kaddr, len) (copyin(uaddr, kaddr, len) ? KERN_MEMORY_ERROR : KERN_SUCCESS)
#define mach_copyout(kaddr, uaddr, len) (copyout(kaddr, uaddr, len) ? KERN_MEMORY_ERROR : KERN_SUCCESS)
#endif /* XNU_KERNEL_PRIVATE */

#else
OS_ALWAYS_INLINE OS_WARN_RESULT static inline int
__copyin_chk(const user_addr_t uaddr, void *__sized_by(len) kaddr, size_t len, size_t chk_size)
{
	if (chk_size < len) {
		panic("__copyin_chk object size check failed: uaddr %p, kaddr %p, (%zu < %zu)", (void*)uaddr, kaddr, len, chk_size);
	}
	return copyin(uaddr, kaddr, len);
}

OS_ALWAYS_INLINE static inline int
__copyout_chk(const void *__sized_by(len) kaddr, user_addr_t uaddr, size_t len, size_t chk_size)
{
	if (chk_size < len) {
		panic("__copyout_chk object size check failed: uaddr %p, kaddr %p, (%zu < %zu)", (void*)uaddr, kaddr, len, chk_size);
	}
	return copyout(kaddr, uaddr, len);
}
#define copyin(uaddr, kaddr, len) __copyin_chk(uaddr, kaddr, len, __builtin_object_size(kaddr, 0))
#define copyout(kaddr, uaddr, len) __copyout_chk(kaddr, uaddr, len, __builtin_object_size(kaddr, 0))

#ifdef XNU_KERNEL_PRIVATE
/* copyio wrappers that return mach error code */
#define mach_copyin(uaddr, kaddr, len) (__copyin_chk(uaddr, kaddr, len, __builtin_object_size(kaddr, 0)) ? KERN_MEMORY_ERROR : KERN_SUCCESS)
#define mach_copyout(kaddr, uaddr, len) (__copyout_chk(kaddr, uaddr, len, __builtin_object_size(kaddr, 0)) ? KERN_MEMORY_ERROR : KERN_SUCCESS)
#endif /* XNU_KERNEL_PRIVATE */

#endif
__END_DECLS
#endif /* _LIBKERN_COPYIO_H_ */
