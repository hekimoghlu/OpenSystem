/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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
 * Glenn Fowler
 * AT&T Research
 *
 * touch file access and modify times of file
 * if flags&PATH_TOUCH_CREATE then file will be created if it doesn't exist
 * if flags&PATH_TOUCH_VERBATIM then times are taken verbatim
 * times have one second granularity
 *
 *	(time_t)(-1)	retain old time
 *	0		use current time
 *
 * the old interface flag values were:
 *	 1	PATH_TOUCH_CREATE
 *	-1	PATH_TOUCH_CREATE|PATH_TOUCH_VERBATIM
 *		PATH_TOUCH_VERBATIM -- not supported
 */

#include <ast.h>
#include <times.h>
#include <tv.h>

int
touch(const char* path, time_t at, time_t mt, int flags)
{
	Tv_t	av;
	Tv_t	mv;
	Tv_t*	ap;
	Tv_t*	mp;

	if (at == (time_t)(-1) && !(flags & PATH_TOUCH_VERBATIM))
		ap = TV_TOUCH_RETAIN;
	else if (!at && !(flags & PATH_TOUCH_VERBATIM))
		ap = 0;
	else
	{
		av.tv_sec = at;
		av.tv_nsec = 0;
		ap = &av;
	}
	if (mt == (time_t)(-1) && !(flags & PATH_TOUCH_VERBATIM))
		mp = TV_TOUCH_RETAIN;
	else if (!mt && !(flags & PATH_TOUCH_VERBATIM))
		mp = 0;
	else
	{
		mv.tv_sec = mt;
		mv.tv_nsec = 0;
		mp = &mv;
	}
	return tvtouch(path, ap, mp, NiL, flags & 1);
}
