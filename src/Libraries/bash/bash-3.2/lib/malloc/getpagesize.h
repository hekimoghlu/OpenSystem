/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 11, 2023.
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
#if defined (HAVE_UNISTD_H)
#  ifdef _MINIX
#    include <sys/types.h>
#  endif
#  include <unistd.h>
#  if defined (_SC_PAGESIZE)
#    define getpagesize() sysconf(_SC_PAGESIZE)
#  else
#    if defined (_SC_PAGE_SIZE)
#      define getpagesize() sysconf(_SC_PAGE_SIZE)
#    endif /* _SC_PAGE_SIZE */
#  endif /* _SC_PAGESIZE */
#endif

#if !defined (getpagesize)
#  ifndef _MINIX
#    include <sys/param.h>
#  endif
#  if defined (PAGESIZE)
#     define getpagesize() PAGESIZE
#  else /* !PAGESIZE */
#    if defined (EXEC_PAGESIZE)
#      define getpagesize() EXEC_PAGESIZE
#    else /* !EXEC_PAGESIZE */
#      if defined (NBPG)
#        if !defined (CLSIZE)
#          define CLSIZE 1
#        endif /* !CLSIZE */
#        define getpagesize() (NBPG * CLSIZE)
#      else /* !NBPG */
#        if defined (NBPC)
#          define getpagesize() NBPC
#        endif /* NBPC */
#      endif /* !NBPG */
#    endif /* !EXEC_PAGESIZE */
#  endif /* !PAGESIZE */
#endif /* !getpagesize */

#if !defined (getpagesize)
#  define getpagesize() 4096  /* Just punt and use reasonable value */
#endif
