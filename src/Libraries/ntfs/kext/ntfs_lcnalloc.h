/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#ifndef _OSX_NTFS_LCNALLOC_H
#define _OSX_NTFS_LCNALLOC_H

#include <sys/errno.h>

#include <kern/locks.h>

#include "ntfs_attr.h"
#include "ntfs_inode.h"
#include "ntfs_runlist.h"
#include "ntfs_types.h"
#include "ntfs_volume.h"

typedef enum {
	FIRST_ZONE	= 0,	/* For sanity checking. */
	MFT_ZONE	= 0,	/* Allocate from $MFT zone. */
	DATA_ZONE	= 1,	/* Allocate from $DATA zone. */
	LAST_ZONE	= 1,	/* For sanity checking. */
} NTFS_CLUSTER_ALLOCATION_ZONES;

__private_extern__ errno_t ntfs_cluster_alloc(ntfs_volume *vol,
		const VCN start_vcn, const s64 count, const LCN start_lcn,
		const NTFS_CLUSTER_ALLOCATION_ZONES zone,
		const BOOL is_extension, ntfs_runlist *runlist);

__private_extern__ errno_t ntfs_cluster_free_from_rl(ntfs_volume *vol,
		ntfs_rl_element *rl, const VCN start_vcn, s64 count,
		s64 *nr_freed);
__private_extern__ errno_t ntfs_cluster_free(ntfs_inode *ni,
		const VCN start_vcn, s64 count, ntfs_attr_search_ctx *ctx,
		s64 *nr_freed);

#endif /* !_OSX_NTFS_LCNALLOC_H */
