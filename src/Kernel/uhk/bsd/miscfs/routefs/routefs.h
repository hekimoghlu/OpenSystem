/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#ifndef _MISCFS_ROUTEFS_DEVFS_H_
#define _MISCFS_ROUTEFS_DEVFS_H_

#include <sys/appleapiopts.h>


__BEGIN_DECLS


#ifdef BSD_KERNEL_PRIVATE

struct routefs_args {
	char    route_path[MAXPATHLEN];/* path name of the target route */
	vnode_t route_rvp; /* vnode of the target of route */
};

struct routefs_mount {
	char    route_path[MAXPATHLEN];/* path name of the target route */
	mount_t route_mount;
	vnode_t route_rvp; /* vnode of the target of route */
	int route_vpvid; /* vnode of the target of route */
};


/*
 * Function: routefs_kernel_mount
 *
 * Purpose:
 *   mount routefs
 *   any links created with devfs_make_link().
 */
int     routefs_kernel_mount(char * routepath);

#endif /* BSD_KERNEL_PRIVATE */

__END_DECLS


#endif /* !_MISCFS_ROUTEFS_DEVFS_H_ */
