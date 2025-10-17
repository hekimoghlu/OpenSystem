/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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

//
//  bc_utils.h
//  BootCache
//
//  Created by Brian Tearse-Doyle on 3/27/17.
//

#ifndef bc_utils_h
#define bc_utils_h

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

void bc_get_volume_info(dev_t volume_dev,
						uint32_t * _Nullable fs_flags_out, // Possible flags are defined in BootCache_private.h
						dev_t * _Nullable apfs_container_dev_out,
						uuid_t _Nullable apfs_container_uuid_out, // filled with uuid_is_null if no container container
						bool * _Nullable apfs_container_has_encrypted_volumes_out,
						bool * _Nullable apfs_container_has_rolling_volumes_out);

void bc_get_group_uuid_for_dev(dev_t dev, uuid_t _Nonnull group_uuid_out);

// Get device string to pass to vnode_lookup for APFS container device
void lookup_dev_name(dev_t dev, char* _Nonnull name, int nmlen);

// Returns an inode for the given vnode, or 0 if unknown
u_int64_t apfs_get_inode(vnode_t _Nonnull vp, u_int64_t lblkno, u_int64_t size, vfs_context_t _Nonnull vfs_context);

#ifdef __cplusplus
}
#endif

#endif /* bc_utils_h */
