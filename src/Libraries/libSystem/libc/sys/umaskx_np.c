/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
#include <sys/acl.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

extern int __umask_extended(int, acl_t);

int
umaskx_np(filesec_t fsec)
{
	acl_t acl = NULL;
	size_t size = 0;
	mode_t newmask = 0;

	if (fsec)
	{
		if ((filesec_get_property(fsec, FILESEC_MODE, &newmask) != 0) && (errno != ENOENT))
			return(-1);

		if (((filesec_get_property(fsec, FILESEC_ACL_RAW, &acl) != 0) ||
			(filesec_get_property(fsec, FILESEC_ACL_ALLOCSIZE, &size) != 0)) &&
		    (errno != ENOENT))
			return(-1);
		if (size == 0)
			acl = NULL;
	}
	return __umask_extended(newmask, acl);
}
