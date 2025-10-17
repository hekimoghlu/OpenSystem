/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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
#include <_types.h>
#include <sys/_types/_size_t.h>

__BEGIN_DECLS

void	*malloc(size_t __size) __result_use_check __alloc_size(1);
void	*calloc(size_t __count, size_t __size) __result_use_check __alloc_size(1,2);
void	 free(void *);
void	*realloc(void *__ptr, size_t __size) __result_use_check __alloc_size(2);
#if !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
void	*valloc(size_t) __alloc_size(1);
#endif // !defined(_ANSI_SOURCE) && (!defined(_POSIX_C_SOURCE) || defined(_DARWIN_C_SOURCE))
#if (__DARWIN_C_LEVEL >= __DARWIN_C_FULL) || \
    (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || \
    (defined(__cplusplus) && __cplusplus >= 201703L)
void    *aligned_alloc(size_t __alignment, size_t __size) __result_use_check __alloc_size(2) __OSX_AVAILABLE(10.15) __IOS_AVAILABLE(13.0) __TVOS_AVAILABLE(13.0) __WATCHOS_AVAILABLE(6.0);
#endif
int 	 posix_memalign(void **__memptr, size_t __alignment, size_t __size) __OSX_AVAILABLE_STARTING(__MAC_10_6, __IPHONE_3_0);

__END_DECLS

#endif /* _MALLOC_UNDERSCORE_MALLOC_H_ */
