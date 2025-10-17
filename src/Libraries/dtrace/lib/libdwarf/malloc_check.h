/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
/* malloc_check.h */

/* A simple libdwarf-aware malloc checker. 
   define WANT_LIBBDWARF_MALLOC_CHECK and rebuild libdwarf
   do make a checking-for-alloc-mistakes libdwarf.
   NOT  recommended for production use.

   When defined, also add malloc_check.c to the list of
   files in Makefile.
*/

#undef WANT_LIBBDWARF_MALLOC_CHECK 
/*#define WANT_LIBBDWARF_MALLOC_CHECK  1 */

#ifdef WANT_LIBBDWARF_MALLOC_CHECK

void dwarf_malloc_check_alloc_data(void * addr,unsigned char code);
void dwarf_malloc_check_dealloc_data(void * addr,unsigned char code);
void dwarf_malloc_check_complete(char *wheremsg); /* called at exit of app */

#else /* !WANT_LIBBDWARF_MALLOC_CHECK */

#define dwarf_malloc_check_alloc_data(a,b)  /* nothing */
#define dwarf_malloc_check_dealloc_data(a,b)  /* nothing */
#define dwarf_malloc_check_complete(a) /* nothing */

#endif /* WANT_LIBBDWARF_MALLOC_CHECK */
