/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
 * Time_t conversion support
 */

#include <tmx.h>
#include <tv.h>

/*
 * touch path <atime,mtime,ctime>
 * (flags&PATH_TOUCH_VERBATIM) treats times verbatim, otherwise:
 * Time_t==0		current time
 * Time_t==TMX_NOTIME	retains path value
 */

int
tmxtouch(const char* path, Time_t at, Time_t mt, Time_t ct, int flags)
{
	Tv_t	av;
	Tv_t	mv;
	Tv_t	cv;
	Tv_t*	ap;
	Tv_t*	mp;
	Tv_t*	cp;

	if (at == TMX_NOTIME && !(flags & PATH_TOUCH_VERBATIM))
		ap = TV_TOUCH_RETAIN;
	else if (!at && !(flags & PATH_TOUCH_VERBATIM))
		ap = 0;
	else
	{
		av.tv_sec = tmxsec(at);
		av.tv_nsec = tmxnsec(at);
		ap = &av;
	}
	if (mt == TMX_NOTIME && !(flags & PATH_TOUCH_VERBATIM))
		mp = TV_TOUCH_RETAIN;
	else if (!mt && !(flags & PATH_TOUCH_VERBATIM))
		mp = 0;
	else
	{
		mv.tv_sec = tmxsec(mt);
		mv.tv_nsec = tmxnsec(mt);
		mp = &mv;
	}
	if (ct == TMX_NOTIME && !(flags & PATH_TOUCH_VERBATIM))
		cp = TV_TOUCH_RETAIN;
	else if (!ct && !(flags & PATH_TOUCH_VERBATIM))
		cp = 0;
	else
	{
		cv.tv_sec = tmxsec(ct);
		cv.tv_nsec = tmxnsec(ct);
		cp = &cv;
	}
	return tvtouch(path, ap, mp, cp, flags & 1);
}
