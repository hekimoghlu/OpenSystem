/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#include <unistd.h>
#include <errno.h>

int
setregid(rgid, egid)
	gid_t rgid, egid;
{
	static gid_t savedgid = -1;
	
	if (savedgid == -1)
		savedgid = getegid();
	/*
	 * we assume that the intent here is to be able to
	 * get back rgid priviledge. So we make sure that
	 * we will be able to do so, but do not actually
	 * set the rgid.
	 */
	if (rgid != -1 && rgid != getgid() && rgid != savedgid) {
		errno = EPERM;
		return (-1);
	}
	if (egid != -1 && setegid(egid) < 0)
		return (-1);
	return (0);
}
