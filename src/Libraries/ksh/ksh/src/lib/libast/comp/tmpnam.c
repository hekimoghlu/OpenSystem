/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
 * tmpnam implementation
 */

#define tmpnam		______tmpnam

#include <ast.h>
#include <stdio.h>

#undef	tmpnam

#undef	_def_map_ast
#include <ast_map.h>

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

#ifndef L_tmpnam
#define L_tmpnam	25
#endif

extern char*
tmpnam(char* s)
{
	static char	buf[L_tmpnam];

	return pathtemp(s ? s : buf, L_tmpnam, NiL, "tn", NiL);
}
