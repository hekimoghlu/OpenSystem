/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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
#ifndef _OSX_NTFS_RUNLIST_H
#define _OSX_NTFS_RUNLIST_H

#include <sys/errno.h>

#include <kern/locks.h>

#include "ntfs_types.h"

/* These definitions need to be before any of the other ntfs_*.h includes. */

/**
 * ntfs_rl_element - in memory vcn to lcn mapping array element
 * @vcn:	starting vcn of the current array element
 * @lcn:	starting lcn of the current array element
 * @length:	length in clusters of the current array element
 *
 * The last vcn (in fact the last vcn + 1) is reached when length == 0.
 *
 * When lcn == -1 this means that the count vcns starting at vcn are not
 * physically allocated (i.e. this is a hole / data is sparse).
 */
typedef struct { /* In memory vcn to lcn mapping structure element. */
	VCN vcn;	/* vcn = Starting virtual cluster number. */
	LCN lcn;	/* lcn = Starting logical cluster number. */
	s64 length;	/* Run length in clusters. */
} ntfs_rl_element;

/**
 * ntfs_runlist - in memory vcn to lcn mapping array including a read/write lock
 * @rl:		pointer to an array of runlist elements
 * @elements:	number of runlist elements in runlist
 * @alloc:	number of bytes allocated for this runlist in memory
 * @lock:	read/write lock for serializing access to @rl
 *
 * This is the runlist structure.  It describes the mapping from file offsets
 * (described as virtual cluster numbers (VCNs)) to on-disk offsets (described
 * as logical cluster numbers (LCNs)).
 *
 * The runlist is made up of an array of runlist elements where each element
 * contains the VCN at which that run starts, the corresponding physical
 * location, i.e. the LCN, at which that run starts and the length in clusters
 * of this run.
 *
 * When doing lookups in the runlist it must be locked for either reading or
 * writing.
 *
 * When modifying the runlist in memory it must be locked for writing.
 *
 * Note that the complete runlist can be spread out over several NTFS
 * attribute fragments in which case only one or only a few parts of the
 * runlist may be mapped at any point in  time.  In this case the regions that
 * are not mapped have placeholders with an LCN of LCN_RL_NOT_MAPPED.  In this
 * case a lookup can lead to a readlocked runlist being writelocked because the
 * in-memory runlist will need updating with the mapped in runlist fragment.
 *
 * Another special value is LCN_HOLE which means that the clusters are not
 * allocated on disk, i.e. this run is sparse, i.e. it is a hole on the
 * attribute.  Thus on reading you just need to interpret the whole run as
 * containing zeroes and on writing you need to allocate real clusters and then
 * write to them.
 *
 * For other special values of LCNs please see below, where the enum
 * LCN_SPECIAL_VALUES is defined.
 */
typedef struct {
	ntfs_rl_element *rl;
	unsigned elements;
	unsigned alloc_count;
	lck_rw_t lock;
} ntfs_runlist;

#include "ntfs.h"
#include "ntfs_layout.h"
#include "ntfs_volume.h"

static inline void ntfs_rl_init(ntfs_runlist *rl)
{
	rl->rl = NULL;
	rl->alloc_count = rl->elements = 0;
	lck_rw_init(&rl->lock, ntfs_lock_grp, ntfs_lock_attr);
}

static inline void ntfs_rl_deinit(ntfs_runlist *rl)
{
	lck_rw_destroy(&rl->lock, ntfs_lock_grp);
}

/**
 * LCN_SPECIAL_VALUES - special values for lcns inside a runlist
 *
 * LCN_HOLE:		run is not allocated on disk, i.e. it is a hole
 * LCN_RL_NOT_MAPPED:	runlist for region starting at current vcn is not
 * 			mapped into memory at the moment, thus it will need to
 * 			be read in before the real lcn can be determined
 * LCN_ENOENT:		the current vcn is the last vcn (actually the last vcn
 * 			+ 1) of the attribute
 * LCN_ENOMEM:		this is only returned in the case of an out of memory
 * 			condition whilst trying to map a runlist fragment for
 * 			example
 * LCN_EIO:		an i/o error occurred when reading/writing the runlist
 */
typedef enum {
	LCN_HOLE		= -1,	/* Keep this as highest value or die! */
	LCN_RL_NOT_MAPPED	= -2,
	LCN_ENOENT		= -3,
	LCN_ENOMEM		= -4,
	LCN_EIO			= -5,
} LCN_SPECIAL_VALUES;

__private_extern__ errno_t ntfs_rl_merge(ntfs_runlist *dst_runlist,
		ntfs_runlist *src_runlist);

__private_extern__ errno_t ntfs_mapping_pairs_decompress(ntfs_volume *vol,
		const ATTR_RECORD *a, ntfs_runlist *runlist);

__private_extern__ LCN ntfs_rl_vcn_to_lcn(const ntfs_rl_element *rl,
		const VCN vcn, s64 *clusters);

__private_extern__ ntfs_rl_element *ntfs_rl_find_vcn_nolock(
		ntfs_rl_element *rl, const VCN vcn);

__private_extern__ errno_t ntfs_get_size_for_mapping_pairs(
		const ntfs_volume *vol, const ntfs_rl_element *rl,
		const VCN first_vcn, const VCN last_vcn, unsigned *mp_size);

__private_extern__ errno_t ntfs_mapping_pairs_build(const ntfs_volume *vol,
		s8 *dst, const unsigned dst_len, const ntfs_rl_element *rl,
		const VCN first_vcn, const VCN last_vcn, VCN *const stop_vcn);

__private_extern__ errno_t ntfs_rl_truncate_nolock(const ntfs_volume *vol,
		ntfs_runlist *const runlist, const s64 new_length);

__private_extern__ errno_t ntfs_rl_punch_nolock(const ntfs_volume *vol,
		ntfs_runlist *runlist, const VCN start_vcn, const s64 len);

__private_extern__ errno_t ntfs_rl_read(ntfs_volume *vol, ntfs_runlist *rl,
		u8 *dst, const s64 size, const s64 initialized_size);

__private_extern__ errno_t ntfs_rl_write(ntfs_volume *vol, u8 *src,
		const s64 size, ntfs_runlist *runlist, s64 ofs, const s64 cnt);

__private_extern__ errno_t ntfs_rl_set(ntfs_volume *vol,
		const ntfs_rl_element *rl, const u8 val);

__private_extern__ s64 ntfs_rl_get_nr_real_clusters(ntfs_runlist *runlist,
		const VCN start_vcn, s64 cnt);

#endif /* !_OSX_NTFS_RUNLIST_H */
