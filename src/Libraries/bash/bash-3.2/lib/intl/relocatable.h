/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
#ifndef _RELOCATABLE_H
#define _RELOCATABLE_H

/* This can be enabled through the configure --enable-relocatable option.  */
#if ENABLE_RELOCATABLE

/* When building a DLL, we must export some functions.  Note that because
   this is a private .h file, we don't need to use __declspec(dllimport)
   in any case.  */
#if defined _MSC_VER && BUILDING_DLL
# define RELOCATABLE_DLL_EXPORTED __declspec(dllexport)
#else
# define RELOCATABLE_DLL_EXPORTED
#endif

/* Sets the original and the current installation prefix of the package.
   Relocation simply replaces a pathname starting with the original prefix
   by the corresponding pathname with the current prefix instead.  Both
   prefixes should be directory names without trailing slash (i.e. use ""
   instead of "/").  */
extern RELOCATABLE_DLL_EXPORTED void
       set_relocation_prefix (const char *orig_prefix,
			      const char *curr_prefix);

/* Returns the pathname, relocated according to the current installation
   directory.  */
extern const char * relocate (const char *pathname);

/* Memory management: relocate() leaks memory, because it has to construct
   a fresh pathname.  If this is a problem because your program calls
   relocate() frequently, think about caching the result.  */

/* Convenience function:
   Computes the current installation prefix, based on the original
   installation prefix, the original installation directory of a particular
   file, and the current pathname of this file.  Returns NULL upon failure.  */
extern const char * compute_curr_prefix (const char *orig_installprefix,
					 const char *orig_installdir,
					 const char *curr_pathname);

#else

/* By default, we use the hardwired pathnames.  */
#define relocate(pathname) (pathname)

#endif

#endif /* _RELOCATABLE_H */
