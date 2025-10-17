/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
 * mime/mailcap to magic support
 */

#include "mimelib.h"

/*
 * close magic handle
 * done this way so that magic is only pulled in
 * if mimetype() is used
 */

static void
drop(Mime_t* mp)
{
	if (mp->magic)
	{
		magicclose(mp->magic);
		mp->magic = 0;
	}
}

/*
 * return mime type for file
 */

char*
mimetype(Mime_t* mp, Sfio_t* fp, const char* file, struct stat* st)
{
	if (mp->disc->flags & MIME_NOMAGIC)
		return 0;
	if (!mp->magic)
	{
		mp->magicd.version = MAGIC_VERSION;
		mp->magicd.flags = MAGIC_MIME;
		mp->magicd.errorf = mp->disc->errorf;
		if (!(mp->magic = magicopen(&mp->magicd)))
		{
			mp->disc->flags |= MIME_NOMAGIC;
			return 0;
		}
		mp->freef = drop;
		magicload(mp->magic, NiL, 0);
	}
	return magictype(mp->magic, fp, file, st);
}
