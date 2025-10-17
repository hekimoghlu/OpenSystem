/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#define	_DARWIN_C_SOURCE

#include <errno.h>
#include <pwd.h>
#include <stdint.h>
#include <unistd.h>

int32_t getgroupcount(const char *name, gid_t basegid);

/*
 * getgroups extension; not limited by NGROUPS_MAX
 */
int
getgroups(int gidsetsize, gid_t grouplist[])
{
    struct passwd *pw;
    int n;

    if ((pw = getpwuid(getuid())) == NULL) {
	errno = EINVAL;
	return -1;
    }
    if (gidsetsize == 0) {
	if ((n = getgroupcount(pw->pw_name, pw->pw_gid)) == 0) {
	    errno = EINVAL;
	    return -1;
	}
	return n;
    }
    n = gidsetsize;
    if (getgrouplist(pw->pw_name, pw->pw_gid, (int *)grouplist, &n) < 0) {
	errno = EINVAL;
	return -1;
    }
    return n;
}
