/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#include <string.h>
#include <sys/mount.h>
#include <sys/sysctl.h>

#include "sysctl_fsid.h"

int
sysctl_fsid(int op, fsid_t *fsid, void *oldp, size_t *oldlenp, void *newp,
    size_t newlen)
{
	int ctlname[CTL_MAXNAME + 2];
	size_t ctllen;
	const char *sysstr = "vfs.generic.ctlbyfsid";
	struct vfsidctl vc;

	ctllen = CTL_MAXNAME + 2;
	if (sysctlnametomib(sysstr, ctlname, &ctllen) == -1) {
		return -1;
	}
	ctlname[ctllen] = op;

	memset(&vc, 0, sizeof(vc));
	vc.vc_vers = VFS_CTL_VERS1;
	vc.vc_fsid = *fsid;
	vc.vc_ptr = newp;
	vc.vc_len = newlen;
	return sysctl(ctlname, (u_int)ctllen + 1, oldp, oldlenp, &vc,
	           sizeof(vc));
}
