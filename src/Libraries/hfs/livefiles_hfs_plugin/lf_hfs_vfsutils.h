/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#ifndef lf_hfs_vfsutils_h
#define lf_hfs_vfsutils_h

#include "lf_hfs.h"

u_int32_t   BestBlockSizeFit(u_int32_t allocationBlockSize, u_int32_t blockSizeLimit, u_int32_t baseMultiple);
int         hfs_MountHFSPlusVolume(struct hfsmount *hfsmp, HFSPlusVolumeHeader *vhp, off_t embeddedOffset, u_int64_t disksize, bool bFailForDirty);
int         hfs_CollectBtreeStats(struct hfsmount *hfsmp, HFSPlusVolumeHeader *vhp, off_t embeddedOffset, void *args);
int         hfs_ValidateHFSPlusVolumeHeader(struct hfsmount *hfsmp, HFSPlusVolumeHeader *vhp);
int         hfs_start_transaction(struct hfsmount *hfsmp);
int         hfs_end_transaction(struct hfsmount *hfsmp);
void*       hfs_malloc( size_t size );
void        hfs_free( void* ptr );
void*       hfs_mallocz( size_t size);
int         hfsUnmount( register struct hfsmount *hfsmp);
void        hfs_lock_mount(struct hfsmount *hfsmp);
void        hfs_unlock_mount(struct hfsmount *hfsmp);
int         hfs_systemfile_lock(struct hfsmount *hfsmp, int flags, enum hfs_locktype locktype);
void        hfs_systemfile_unlock(struct hfsmount *hfsmp, int flags);
u_int32_t   hfs_freeblks(struct hfsmount * hfsmp, int wantreserve);
short       MacToVFSError(OSErr err);

void hfs_reldirhint(struct cnode *dcp, directoryhint_t * relhint);
void hfs_insertdirhint(struct cnode *dcp, directoryhint_t * hint);
void hfs_reldirhints(struct cnode *dcp, int stale_hints_only);

directoryhint_t* hfs_getdirhint(struct cnode *dcp, int index, int detach);

int  hfs_systemfile_lock(struct hfsmount *hfsmp, int flags, enum hfs_locktype locktype);
void hfs_systemfile_unlock(struct hfsmount *hfsmp, int flags);
bool overflow_extents(struct filefork *fp);
int  hfs_namecmp(const u_int8_t *str1, size_t len1, const u_int8_t *str2, size_t len2);
int  hfs_strstr(const u_int8_t *str1, size_t len1, const u_int8_t *str2, size_t len2);
int  hfs_apendixcmp(const u_int8_t *str1, size_t len1, const u_int8_t *str2, size_t len2);
void hfs_remove_orphans(struct hfsmount * hfsmp);
int  hfs_erase_unused_nodes(struct hfsmount *hfsmp);

int hfs_lock_global (struct hfsmount *hfsmp, enum hfs_locktype locktype);
void hfs_unlock_global (struct hfsmount *hfsmp);


// Journaing
int hfs_early_journal_init(struct hfsmount *hfsmp, HFSPlusVolumeHeader *vhp,
					   void *_args, off_t embeddedOffset, daddr64_t mdb_offset,
					   HFSMasterDirectoryBlock *mdbp);
errno_t hfs_flush(struct hfsmount *hfsmp, hfs_flush_mode_t mode);

#endif /* lf_hfs_vfsutils_h */
