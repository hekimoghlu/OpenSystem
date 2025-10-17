/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#ifndef _MM_MALLOC_H_INCLUDED
#define _MM_MALLOC_H_INCLUDED

#if defined(__powerpc64__) &&                                                  \
    (defined(__linux__) || defined(__FreeBSD__) || defined(_AIX))

#include <stdlib.h>

/* We can't depend on <stdlib.h> since the prototype of posix_memalign
   may not be visible.  */
#ifndef __cplusplus
extern int posix_memalign(void **, size_t, size_t);
#else
extern "C" int posix_memalign(void **, size_t, size_t);
#endif

static __inline void *_mm_malloc(size_t __size, size_t __alignment) {
  /* PowerPC64 ELF V2 ABI requires quadword alignment.  */
  size_t __vec_align = sizeof(__vector float);
  void *__ptr;

  if (__alignment < __vec_align)
    __alignment = __vec_align;
  if (posix_memalign(&__ptr, __alignment, __size) == 0)
    return __ptr;
  else
    return NULL;
}

static __inline void _mm_free(void *__ptr) { free(__ptr); }

#else
#include_next <mm_malloc.h>
#endif

#endif /* _MM_MALLOC_H_INCLUDED */
