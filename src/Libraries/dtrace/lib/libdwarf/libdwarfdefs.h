/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
/* libdwarfdefs.h
*/

#ifndef LIBDWARFDEFS_H
#define LIBDWARFDEFS_H

/* We want __uint32_t and __uint64_t and __int32_t __int64_t
   properly defined but not duplicated, since duplicate typedefs
   are not legal C.
*/
/*
 HAVE___UINT32_T
 HAVE___UINT64_T will be set by configure if
 our 4 types are predefined in compiler
*/


#if (!defined(HAVE___UINT32_T)) && defined(HAVE___UINT32_T_IN_SGIDEFS_H)
#include <sgidefs.h>		/* sgidefs.h defines them */
#define HAVE___UINT32_T 1
#endif

#if (!defined(HAVE___UINT64_T)) && defined(HAVE___UINT64_T_IN_SGIDEFS_H)
#include <sgidefs.h>		/* sgidefs.h defines them */
#define HAVE___UINT64_T 1
#endif


#if (!defined(HAVE___UINT32_T)) &&   \
	defined(HAVE_SYS_TYPES_H) &&   \
	defined(HAVE___UINT32_T_IN_SYS_TYPES_H)
#  include <sys/types.h>
#define HAVE___UINT32_T 1
#endif

#if (!defined(HAVE___UINT64_T)) &&   \
	defined(HAVE_SYS_TYPES_H) &&   \
	defined(HAVE___UINT64_T_IN_SYS_TYPES_H)
#  include <sys/types.h>
#define HAVE___UINT64_T 1
#endif

#ifndef HAVE___UINT32_T
typedef int __int32_t;
typedef unsigned __uint32_t;
#define HAVE___UINT32_T 1
#endif

#ifndef HAVE___UINT64_T
typedef long long __int64_t;
typedef unsigned long long __uint64_t;
#define HAVE___UINT64_T 1
#endif

#endif /* LIBDWARFDEFS_H */
