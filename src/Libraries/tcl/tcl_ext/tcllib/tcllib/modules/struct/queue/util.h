/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
#ifndef _UTIL_H
#define _UTIL_H 1

#include <tcl.h>

/* Allocation macros for common situations.
 */

#define ALLOC(type)    (type *) ckalloc (sizeof (type))
#define NALLOC(n,type) (type *) ckalloc ((n) * sizeof (type))

/* Assertions in general, and asserting the proper range of an array index.
 */

#undef  QUEUE_DEBUG
#define QUEUE_DEBUG 1

#ifdef QUEUE_DEBUG
#define XSTR(x) #x
#define STR(x) XSTR(x)
#define RANGEOK(i,n) ((0 <= (i)) && (i < (n)))
#define ASSERT(x,msg) if (!(x)) { Tcl_Panic (msg " (" #x "), in file " __FILE__ " @line " STR(__LINE__));}
#define ASSERT_BOUNDS(i,n) ASSERT (RANGEOK(i,n),"array index out of bounds: " STR(i) " > " STR(n))
#else
#define ASSERT(x,msg)
#define ASSERT_BOUNDS(i,n)
#endif

#endif /* _UTIL_H */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
