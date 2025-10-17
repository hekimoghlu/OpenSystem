/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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
#ifndef _NTFS_LCNALLOC_H
#define _NTFS_LCNALLOC_H

#include "types.h"
#include "runlist.h"
#include "volume.h"

/**
 * enum NTFS_CLUSTER_ALLOCATION_ZONES -
 */
typedef enum {
	FIRST_ZONE	= 0,	/* For sanity checking. */
	MFT_ZONE	= 0,	/* Allocate from $MFT zone. */
	DATA_ZONE	= 1,	/* Allocate from $DATA zone. */
	LAST_ZONE	= 1,	/* For sanity checking. */
} NTFS_CLUSTER_ALLOCATION_ZONES;

extern runlist *ntfs_cluster_alloc(ntfs_volume *vol, VCN start_vcn, s64 count,
		LCN start_lcn, const NTFS_CLUSTER_ALLOCATION_ZONES zone);

extern int ntfs_cluster_free_from_rl(ntfs_volume *vol, runlist *rl);
extern int ntfs_cluster_free_basic(ntfs_volume *vol, s64 lcn, s64 count);

extern int ntfs_cluster_free(ntfs_volume *vol, ntfs_attr *na, VCN start_vcn,
		s64 count);

#endif /* defined _NTFS_LCNALLOC_H */
