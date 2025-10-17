/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#undef TYQUAD0
#ifdef NO_TYQUAD
#undef TYQUAD
#define TYQUAD_inc 0
#undef NO_LONG_LONG
#define NO_LONG_LONG
#else
#define TYQUAD 5
#define TYQUAD_inc 1
#ifdef NO_LONG_LONG
#define TYQUAD0
#else
#ifndef Llong
typedef long long Llong;
#endif
#ifndef ULlong
typedef unsigned long long ULlong;
#endif
#endif /*NO_LONG_LONG*/
#endif /*NO_TYQUAD*/

#define TYUNKNOWN 0
#define TYADDR 1
#define TYINT1 2
#define TYSHORT 3
#define TYLONG 4
/* #define TYQUAD 5 */
#define TYREAL (5+TYQUAD_inc)
#define TYDREAL (6+TYQUAD_inc)
#define TYCOMPLEX (7+TYQUAD_inc)
#define TYDCOMPLEX (8+TYQUAD_inc)
#define TYLOGICAL1 (9+TYQUAD_inc)
#define TYLOGICAL2 (10+TYQUAD_inc)
#define TYLOGICAL (11+TYQUAD_inc)
#define TYCHAR (12+TYQUAD_inc)
#define TYSUBR (13+TYQUAD_inc)
#define TYERROR (14+TYQUAD_inc)
#define TYCILIST (15+TYQUAD_inc)
#define TYICILIST (16+TYQUAD_inc)
#define TYOLIST (17+TYQUAD_inc)
#define TYCLLIST (18+TYQUAD_inc)
#define TYALIST (19+TYQUAD_inc)
#define TYINLIST (20+TYQUAD_inc)
#define TYVOID (21+TYQUAD_inc)
#define TYLABEL (22+TYQUAD_inc)
#define TYFTNLEN (23+TYQUAD_inc)
/* TYVOID is not in any tables. */

/* NTYPES, NTYPES0 -- Total number of types, used to allocate tables indexed by
   type.  Such tables can include the size (in bytes) of objects of a given
   type, or labels for returning objects of different types from procedures
   (see array   rtvlabels)   */

#define NTYPES TYVOID
#define NTYPES0 TYCILIST
#define TYBLANK TYSUBR		/* Huh? */

