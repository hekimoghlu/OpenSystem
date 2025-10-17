/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#ifndef _LOCKD_MACH_TYPES_H_
#define _LOCKD_MACH_TYPES_H_

/*
 * XXX NFSV3_MAX_FH_SIZE is defined in sys/mount.h, but we can't include
 * that here. Osfmk includes libsa/types.h which causes massive conflicts
 * with sys/types.h that get indirectly included with sys/mount.h. In user
 * land below will work on a build that does not yet have the new macro
 * definition.
 */

#ifndef NFSV3_MAX_FH_SIZE
#define NFSV3_MAX_FH_SIZE 64
#endif

typedef uint32_t sock_storage[32];
typedef uint32_t xcred[19];
typedef uint8_t nfs_handle[NFSV3_MAX_FH_SIZE];

#endif
