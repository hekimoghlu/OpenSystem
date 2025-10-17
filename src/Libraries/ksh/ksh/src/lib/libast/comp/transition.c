/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
 * transient code to aid transition between releases
 */

#include <ast.h>

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

#define STUB		1

/*
 * 2006-09-28
 *
 *	on some systems the _std_strtol iffe changed (due to a faulty
 *	test prototype) and the cause programs dynamically linked to
 *	an updated -last to fail at runtime with missing _ast_strtol etc.
 */

#if !_std_strtol

#ifndef strtol
#undef	STUB
extern long
_ast_strtol(const char* a, char** b, int c)
{
	return strtol(a, b, c);
}
#endif

#ifndef strtoul
#undef	STUB
extern unsigned long
_ast_strtoul(const char* a, char** b, int c)
{
	return strtoul(a, b, c);
}
#endif

#ifndef strtoll
#undef	STUB
extern intmax_t
_ast_strtoll(const char* a, char** b, int c)
{
	return strtoll(a, b, c);
}
#endif

#ifndef strtoull
#undef	STUB
extern uintmax_t
_ast_strtoull(const char* a, char** b, int c)
{
	return strtoull(a, b, c);
}
#endif

#endif

#if STUB
NoN(transition)
#endif
