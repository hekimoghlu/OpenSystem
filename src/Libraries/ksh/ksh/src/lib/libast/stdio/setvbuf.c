/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
setvbuf(Sfio_t* f, char* buf, int type, size_t size)
{
	STDIO_INT(f, "setvbuf", int, (Sfio_t*, char*, int, size_t), (f, buf, type, size))

	if (type == _IOLBF)
		sfset(f, SF_LINE, 1);
	else if (f->flags & SF_STRING)
		return -1;
	else if (type == _IONBF)
	{	
		sfsync(f);
		sfsetbuf(f, NiL, 0);
	}
	else if (type == _IOFBF)
	{	
		if (size == 0)
			size = SF_BUFSIZE;
		sfsync(f);
		sfsetbuf(f, (Void_t*)buf, size);
	}
	return 0;
}
