/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/attr.h>
#include <sys/time.h>
#include <unistd.h>
#include <strings.h>

int
lutimes(const char *path, const struct timeval *times)
{
	struct stat s;
	struct attrlist a;
	struct {
	    struct timespec mod;
	    struct timespec access;
	} t;

	if(lstat(path, &s) < 0)
		return -1;
	if((s.st_mode & S_IFMT) != S_IFLNK)
		return utimes(path, times);
	bzero(&a, sizeof(a));
	a.bitmapcount = ATTR_BIT_MAP_COUNT;
	a.commonattr = ATTR_CMN_MODTIME | ATTR_CMN_ACCTIME;
	if(times) {
		TIMEVAL_TO_TIMESPEC(&times[0], &t.access);
		TIMEVAL_TO_TIMESPEC(&times[1], &t.mod);
	} else {
		struct timeval now;

		if(gettimeofday(&now, NULL) < 0)
			return -1;
		TIMEVAL_TO_TIMESPEC(&now, &t.access);
		TIMEVAL_TO_TIMESPEC(&now, &t.mod);
	}
	return setattrlist(path, &a, &t, sizeof(t), FSOPT_NOFOLLOW);
}
