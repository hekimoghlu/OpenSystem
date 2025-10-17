/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#include <unistd.h>
#include <strings.h>

int
lchflags(const char *path, unsigned int flags)
{
	struct stat s;
	struct attrlist a;

	if(lstat(path, &s) < 0)
		return -1;
	if((s.st_mode & S_IFMT) != S_IFLNK)
		return chflags(path, flags);
	bzero(&a, sizeof(a));
	a.bitmapcount = ATTR_BIT_MAP_COUNT;
	a.commonattr = ATTR_CMN_FLAGS;
	return setattrlist(path, &a, &flags, sizeof(unsigned int), FSOPT_NOFOLLOW);
}
