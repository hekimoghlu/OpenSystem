/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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
#ifndef _UAPI_LINUX_MEMPOLICY_H
#define _UAPI_LINUX_MEMPOLICY_H
#include <linux/errno.h>
enum {
  MPOL_DEFAULT,
  MPOL_PREFERRED,
  MPOL_BIND,
  MPOL_INTERLEAVE,
  MPOL_LOCAL,
  MPOL_PREFERRED_MANY,
  MPOL_WEIGHTED_INTERLEAVE,
  MPOL_MAX,
};
#define MPOL_F_STATIC_NODES (1 << 15)
#define MPOL_F_RELATIVE_NODES (1 << 14)
#define MPOL_F_NUMA_BALANCING (1 << 13)
#define MPOL_MODE_FLAGS (MPOL_F_STATIC_NODES | MPOL_F_RELATIVE_NODES | MPOL_F_NUMA_BALANCING)
#define MPOL_F_NODE (1 << 0)
#define MPOL_F_ADDR (1 << 1)
#define MPOL_F_MEMS_ALLOWED (1 << 2)
#define MPOL_MF_STRICT (1 << 0)
#define MPOL_MF_MOVE (1 << 1)
#define MPOL_MF_MOVE_ALL (1 << 2)
#define MPOL_MF_LAZY (1 << 3)
#define MPOL_MF_INTERNAL (1 << 4)
#define MPOL_MF_VALID (MPOL_MF_STRICT | MPOL_MF_MOVE | MPOL_MF_MOVE_ALL)
#define MPOL_F_SHARED (1 << 0)
#define MPOL_F_MOF (1 << 3)
#define MPOL_F_MORON (1 << 4)
#define RECLAIM_ZONE (1 << 0)
#define RECLAIM_WRITE (1 << 1)
#define RECLAIM_UNMAP (1 << 2)
#endif
