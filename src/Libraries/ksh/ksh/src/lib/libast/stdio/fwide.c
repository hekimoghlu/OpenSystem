/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
fwide(Sfio_t* f, int mode)
{
	STDIO_INT(f, "fwide", int, (Sfio_t*, int), (f, mode))

	if (mode > 0)
	{
		f->bits &= ~SF_MB;
		f->bits |= SF_WC;
	}
	else if (mode < 0)
	{
		f->bits &= ~SF_WC;
		f->bits |= SF_MB;
	}
	if (f->bits & SF_MB)
		return -1;
	if (f->bits & SF_WC)
		return 1;
	if ((f->flags & SF_SYNCED) || f->next > f->data)
	{
		f->bits |= SF_MB;
		return -1;
	}
	return 0;
}
