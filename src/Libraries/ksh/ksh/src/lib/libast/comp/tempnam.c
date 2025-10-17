/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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
#pragma prototyped
/*
 * tempnam implementation
 */

#include <ast_std.h>

#ifdef tempnam
#define _def_tempnam	1
#else
#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:hide tempnam
#else
#define tempnam		______tempnam
#endif
#endif

#include <ast.h>
#include <stdio.h>

#if !_def_tempnam
#if defined(__STDPP__directive) && defined(__STDPP__hide)
__STDPP__directive pragma pp:nohide tempnam
#else
#undef	tempnam
#endif
#endif

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern char*
tempnam(const char* dir, const char* pfx)
{
	return pathtmp(NiL, dir, pfx, NiL);
}
