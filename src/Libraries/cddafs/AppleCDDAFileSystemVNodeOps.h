/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
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
// AppleCDDAFileSystemVNodeOps.h created by CJS on Mon 13-Apr-2000

#ifndef __APPLE_CDDA_FS_VNODE_OPS_H__
#define __APPLE_CDDA_FS_VNODE_OPS_H__

#include <sys/param.h>

#ifdef __cplusplus
extern "C" {
#endif


// BlockSize constants
enum
{
	kPhysicalMediaBlockSize		= 2352,
	kMaxBlocksPerRead			= MAXBSIZE / kPhysicalMediaBlockSize,			// Max blocks to read per bread()
	kMaxBytesPerRead			= kMaxBlocksPerRead * kPhysicalMediaBlockSize	// Max bytes to read per bread()
};

#ifdef __cplusplus
}
#endif

#endif // __APPLE_CDDA_FS_VNODE_OPS_H__


//-----------------------------------------------------------------------------
//				End				Of			File
//-----------------------------------------------------------------------------
