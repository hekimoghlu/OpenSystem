/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#ifndef _OS_OSLIB_H
#define _OS_OSLIB_H

#include <libkern/OSBase.h>

#ifdef KERNEL
#define MACH_ASSERT 1
#endif

__BEGIN_DECLS

#include <stdarg.h>
#include <sys/systm.h>

#include <kern/assert.h>
#ifdef KERNEL_PRIVATE
#include <kern/kalloc.h>
#endif

__END_DECLS


#if XNU_KERNEL_PRIVATE
#include <libkern/OSAtomic.h>
#include <libkern/c++/OSCPPDebug.h>

#define kallocp_type_container(ty, countp, flags) ({                           \
	uint32_t *__countp = (countp);                                         \
	struct kalloc_result __kar;                                            \
	static KALLOC_TYPE_VAR_DEFINE_3(kt_view_var, ty, KT_SHARED_ACCT);      \
	__kar = kalloc_ext(kt_mangle_var_view(kt_view_var),                    \
	    kt_size(0, sizeof(ty), *__countp),                                 \
	    Z_VM_TAG_BT(flags | Z_FULLSIZE | Z_SPRAYQTN | Z_SET_NOTEARLY,     \
	    VM_KERN_MEMORY_LIBKERN), NULL);                                    \
	*__countp = (uint32_t)MIN(__kar.size / sizeof(ty), UINT32_MAX);        \
	(ty *)__kar.addr;                                                      \
})

#define kreallocp_type_container(ty, ptr, old_count, countp, flags) ({         \
	uint32_t *__countp = (countp);                                         \
	struct kalloc_result __kar;                                            \
	static KALLOC_TYPE_VAR_DEFINE_3(kt_view_var, ty, KT_SHARED_ACCT);      \
	__kar = krealloc_ext(kt_mangle_var_view(kt_view_var), ptr,             \
	    kt_size(0, sizeof(ty), old_count),                                 \
	    kt_size(0, sizeof(ty), *__countp),                                 \
	    Z_VM_TAG_BT(flags | Z_FULLSIZE | Z_SPRAYQTN | Z_SET_NOTEARLY,     \
	    VM_KERN_MEMORY_LIBKERN), NULL);                                    \
	*__countp = (uint32_t)MIN(__kar.size / sizeof(ty), UINT32_MAX);        \
	(ty *)__kar.addr;                                                      \
})

#if OSALLOCDEBUG

#if IOTRACKING
#define OSCONTAINER_ACCUMSIZE(s) do { OSAddAtomicLong((s), &debug_container_malloc_size); trackingAccumSize(s); } while(0)
#else
#define OSCONTAINER_ACCUMSIZE(s) do { OSAddAtomicLong((s), &debug_container_malloc_size); } while(0)
#endif
#define OSMETA_ACCUMSIZE(s)      do { OSAddAtomicLong((s), &debug_container_malloc_size); } while(0)
#define OSIVAR_ACCUMSIZE(s)      do { OSAddAtomicLong((s), &debug_ivars_size);            } while(0)

#else /* OSALLOCDEBUG */

#define OSCONTAINER_ACCUMSIZE(s)
#define OSMETA_ACCUMSIZE(s)
#define OSIVAR_ACCUMSIZE(s)

#endif  /* !OSALLOCDEBUG */
#endif  /* XNU_KERNEL_PRIVATE */

#ifndef NULL
#if defined (__cplusplus)
#if __cplusplus >= 201103L
#define NULL nullptr
#else
#define NULL 0
#endif
#else
#define NULL ((void *)0)
#endif
#endif

#endif /* _OS_OSLIB_H  */
