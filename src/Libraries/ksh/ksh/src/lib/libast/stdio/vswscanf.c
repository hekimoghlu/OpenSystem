/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

#include "stdhdr.h"

int
vswscanf(const wchar_t* s, const wchar_t* fmt, va_list args)
{
	Sfio_t	f;

	if (!s)
		return -1;

	/*
	 * make a fake stream
	 */

	SFCLEAR(&f, NiL);
	f.flags = SF_STRING|SF_READ;
	f.bits = SF_PRIVATE;
	f.mode = SF_READ;
	f.size = wcslen(s) * sizeof(wchar_t);
	f.data = f.next = f.endw = (uchar*)s;
	f.endb = f.endr = f.data + f.size;

	/*
	 * sfio does the rest
	 */

	return vfwscanf(&f, fmt, args);
}
