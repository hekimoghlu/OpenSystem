/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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
// AppleCDDAFileSystemVFSOps.h created by CJS on Mon 27-Apr-2000

#ifndef __APPLE_CDDA_FS_VFS_OPS_H__
#define __APPLE_CDDA_FS_VFS_OPS_H__

#ifdef __cplusplus
extern "C" {
#endif

// Project Includes
#ifndef __APPLE_CDDA_FS_VNODE_OPS_H__
#include "AppleCDDAFileSystemVNodeOps.h"
#endif

#ifdef KERNEL

#include <sys/types.h>
#include <sys/ucred.h>
#include <sys/mount.h>
#include <sys/vnode.h>

#endif


//-----------------------------------------------------------------------------
//	Constants
//-----------------------------------------------------------------------------

enum
{
	kAppleCDDARootFileID	= 2,
	kNumberOfFakeDirEntries	= 3,
	kOffsetForFiles			= 100
};

extern int ( **gCDDA_VNodeOp_p )( void * );


//-----------------------------------------------------------------------------
//	Function Prototypes
//-----------------------------------------------------------------------------


int CDDA_Mount 					( mount_t mountPtr,
								  vnode_t blockDeviceVNodePtr,
								  user_addr_t data,
								  vfs_context_t context );
int CDDA_Unmount				( mount_t mountPtr,
								  int theFlags,
								  vfs_context_t context );
int CDDA_Root					( mount_t mountPtr,
								  vnode_t * vnodeHandle,
								  vfs_context_t context );
int CDDA_VFSGetAttributes		( mount_t mountPtr,
    							  struct vfs_attr * attrPtr,
 								  vfs_context_t context );
int	CDDA_VGet					( mount_t mountPtr,
								  ino64_t  nodeID,
								  vnode_t * vNodeHandle,
								  vfs_context_t context );
extern struct vfsops gCDDA_VFSOps;

// Private internal methods
int
CDDA_VGetInternal ( mount_t 				mountPtr,
					ino64_t  				ino,
					vnode_t					parentVNodePtr,
					struct componentname *	compNamePtr,
					vnode_t * 				vNodeHandle );


#ifdef __cplusplus
}
#endif

#endif // __APPLE_CDDA_FS_VFS_OPS_H__
