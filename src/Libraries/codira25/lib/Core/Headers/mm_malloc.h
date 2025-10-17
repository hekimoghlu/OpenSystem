/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#ifndef __MM_MALLOC_H
#define __MM_MALLOC_H

#include <stdlib.h>

#ifdef _WIN32
#include <malloc.h>
#else
#ifndef __cplusplus
extern int posix_memalign(void **__memptr, size_t __alignment, size_t __size);
#else
// Some systems (e.g. those with GNU libc) declare posix_memalign with an
// exception specifier. Via an "egregious workaround" in
// Sema::CheckEquivalentExceptionSpec, Clang accepts the following as a valid
// redeclaration of glibc's declaration.
extern "C" int posix_memalign(void **__memptr, size_t __alignment, size_t __size);
#endif
#endif

#if !(defined(_WIN32) && defined(_mm_malloc))
static __inline__ void *__attribute__((__always_inline__, __nodebug__,
                                       __malloc__, __alloc_size__(1),
                                       __alloc_align__(2)))
_mm_malloc(size_t __size, size_t __align) {
  if (__align == 1) {
    return malloc(__size);
  }

  if (!(__align & (__align - 1)) && __align < sizeof(void *))
    __align = sizeof(void *);

  void *__mallocedMemory;
#if defined(__MINGW32__)
  __mallocedMemory = __mingw_aligned_malloc(__size, __align);
#elif defined(_WIN32)
  __mallocedMemory = _aligned_malloc(__size, __align);
#else
  if (posix_memalign(&__mallocedMemory, __align, __size))
    return 0;
#endif

  return __mallocedMemory;
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_free(void *__p)
{
#if defined(__MINGW32__)
  __mingw_aligned_free(__p);
#elif defined(_WIN32)
  _aligned_free(__p);
#else
  free(__p);
#endif
}
#endif

#endif /* __MM_MALLOC_H */
