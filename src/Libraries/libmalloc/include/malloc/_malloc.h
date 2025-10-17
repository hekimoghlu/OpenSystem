/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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
#ifndef _MALLOC_UNDERSCORE_MALLOC_H_
#define _MALLOC_UNDERSCORE_MALLOC_H_

/*
 * This header is included from <stdlib.h>, so the contents of this file have
 * broad source compatibility and POSIX conformance implications.
 * Be cautious about what is included and declared here.
 */

#include <Availability.h>
#include <sys/cdefs.h>
#if __has_include(<sys/_types/_size_t.h>)
#include <sys/_types/_size_t.h>
#else
#define __need_size_t
#include <stddef.h>
#undef __need_size_t
#endif

#if !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
#include <malloc/_malloc_type.h>
#else
#define _MALLOC_TYPED(override, type_param_pos)
#endif

#include <malloc/_ptrcheck.h>
__ptrcheck_abi_assume_single()

__BEGIN_DECLS

void * __sized_by_or_null(__size) malloc(size_t __size) __result_use_check __alloc_size(1) _MALLOC_TYPED(malloc_type_malloc, 1);
void * __sized_by_or_null(__count * __size) calloc(size_t __count, size_t __size) __result_use_check __alloc_size(1,2) _MALLOC_TYPED(malloc_type_calloc, 2);
void  free(void * __unsafe_indexable);
void * __sized_by_or_null(__size) realloc(void * __unsafe_indexable __ptr, size_t __size) __result_use_check __alloc_size(2) _MALLOC_TYPED(malloc_type_realloc, 2);
void * __sized_by_or_null(__size) reallocf(void * __unsafe_indexable __ptr, size_t __size) __result_use_check __alloc_size(2);
#if !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
void * __sized_by_or_null(__size) valloc(size_t __size) __result_use_check __alloc_size(1) _MALLOC_TYPED(malloc_type_valloc, 1);
#endif /* !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE)) */
#if (defined(__DARWIN_C_LEVEL) && defined(__DARWIN_C_FULL) && __DARWIN_C_LEVEL >= __DARWIN_C_FULL) || \
    (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || \
    (defined(__cplusplus) && __cplusplus >= 201703L)
void * __sized_by_or_null(__size) aligned_alloc(size_t __alignment, size_t __size) __result_use_check __alloc_align(1) __alloc_size(2) _MALLOC_TYPED(malloc_type_aligned_alloc, 2) __OSX_AVAILABLE(10.15) __IOS_AVAILABLE(13.0) __TVOS_AVAILABLE(13.0) __WATCHOS_AVAILABLE(6.0);
#endif
/* rdar://120689514 */
int   posix_memalign(void * __unsafe_indexable *__memptr, size_t __alignment, size_t __size) _MALLOC_TYPED(malloc_type_posix_memalign, 3)  __OSX_AVAILABLE_STARTING(__MAC_10_6, __IPHONE_3_0);

__END_DECLS

#endif /* _MALLOC_UNDERSCORE_MALLOC_H_ */
