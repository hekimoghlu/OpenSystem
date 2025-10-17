/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#ifndef _NETSMB_SMB_FID_H_
#define	_NETSMB_SMB_FID_H_

#include <netsmb/smb_2.h>

#define	smb_fid_table_lock(share)	(lck_mtx_lock(&(share)->ss_fid_lock))
#define	smb_fid_table_unlock(share)	(lck_mtx_unlock(&(share)->ss_fid_lock))

LIST_HEAD(fid_list_head, fid_node_t);

// An element in the global fid hash table
typedef struct fid_node_t
{
	SMBFID  fid;
	SMB2FID smb2_fid;
	LIST_ENTRY(fid_node_t) link;
	
} SMB_FID_NODE;

typedef struct fid_hash_table_slot
{
	struct fid_list_head fid_list;
	
} FID_HASH_TABLE_SLOT;

// Global fid hash table
#define SMB_FID_TABLE_SIZE 4096
#define SMB_FID_TABLE_MASK 0x0000000000000fff

uint64_t smb_fid_count_all(struct smb_share *share);
void smb_fid_delete_all(struct smb_share *share);
int smb_fid_get_kernel_fid(struct smb_share *share, SMBFID fid, int remove_fid,
                           SMB2FID *smb2_fid);
int smb_fid_get_user_fid(struct smb_share *share, SMB2FID smb2_fid, 
                         SMBFID *fid);
int smb_fid_update_kernel_fid(struct smb_share *share, SMBFID fid,
                              SMB2FID new_smb2_fid);

#endif
