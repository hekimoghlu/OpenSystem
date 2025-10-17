/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 14, 2025.
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
#ifndef _XMALLOC_H
#define _XMALLOC_H 1

void *xmalloc_impl(size_t size, const char *file, int line, const char *func);
void *xcalloc_impl(size_t nmemb, size_t size, const char *file, int line,
		   const char *func);
void xfree_impl(void *ptr, const char *file, int line, const char *func);
void *xrealloc_impl(void *ptr, size_t new_size, const char *file, int line,
		    const char *func);
int xmalloc_dump_leaks(void);
void xmalloc_configure(int fail_after);


#ifndef XMALLOC_INTERNAL
#ifdef MALLOC_DEBUGGING

/* Version 2.4 and later of GCC define a magical variable `__PRETTY_FUNCTION__'
   which contains the name of the function currently being defined.
#  define __XMALLOC_FUNCTION	 __PRETTY_FUNCTION__
   This is broken in G++ before version 2.6.
   C9x has a similar variable called __func__, but prefer the GCC one since
   it demangles C++ function names.  */
# ifdef __GNUC__
#  if __GNUC__ > 2 || (__GNUC__ == 2 \
		       && __GNUC_MINOR__ >= (defined __cplusplus ? 6 : 4))
#   define __XMALLOC_FUNCTION	 __PRETTY_FUNCTION__
#  else
#   define __XMALLOC_FUNCTION	 ((const char *) 0)
#  endif
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __XMALLOC_FUNCTION	 __func__
#  else
#   define __XMALLOC_FUNCTION	 ((const char *) 0)
#  endif
# endif

#define xmalloc(size) xmalloc_impl(size, __FILE_NAME__, __LINE__, \
				   __XMALLOC_FUNCTION)
#define xcalloc(nmemb, size) xcalloc_impl(nmemb, size, __FILE_NAME__, __LINE__, \
					  __XMALLOC_FUNCTION)
#define xfree(ptr) xfree_impl(ptr, __FILE_NAME__, __LINE__, __XMALLOC_FUNCTION)
#define xrealloc(ptr, new_size) xrealloc_impl(ptr, new_size, __FILE_NAME__, \
					      __LINE__, __XMALLOC_FUNCTION)
#undef malloc
#undef calloc
#undef free
#undef realloc

#define malloc	USE_XMALLOC_INSTEAD_OF_MALLOC
#define calloc	USE_XCALLOC_INSTEAD_OF_CALLOC
#define free	USE_XFREE_INSTEAD_OF_FREE
#define realloc USE_XREALLOC_INSTEAD_OF_REALLOC

#else /* !MALLOC_DEBUGGING */

#include <stdlib.h>

#define xmalloc(size) malloc(size)
#define xcalloc(nmemb, size) calloc(nmemb, size)
#define xfree(ptr) free(ptr)
#define xrealloc(ptr, new_size) realloc(ptr, new_size)

#endif /* !MALLOC_DEBUGGING */
#endif /* !XMALLOC_INTERNAL */

#endif /* _XMALLOC_H */

/* EOF */
