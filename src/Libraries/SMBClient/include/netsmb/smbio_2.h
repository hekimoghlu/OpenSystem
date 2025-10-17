/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#ifndef _NETSMB_SMBIO_2_H_
#define _NETSMB_SMBIO_2_H_

#include <netsmb/smbio.h>

struct open_outparm_ex {
	uint64_t createTime;
	uint64_t accessTime;
	uint64_t writeTime;
	uint64_t changeTime;
	uint32_t attributes;
	uint64_t allocationSize;
	uint64_t fileSize;
	uint64_t fileInode;
	uint32_t maxAccessRights;
    SMBFID fid;
};

int smb_is_smb2(struct smb_ctx *ctx);

int smb2io_check_directory(struct smb_ctx *ctx, const void *path,
                           uint32_t flags, uint32_t *nt_error);
int smb2io_close_file(void *smbctx, SMBFID fid);
int smb2io_get_dfs_referral(struct smb_ctx *smbctx, CFStringRef dfs_referral_str,
                            uint16_t max_referral_level,
                            CFMutableDictionaryRef *out_referral_dict);
int smb2io_ioctl(struct smb_ctx *smbctx, SMBFID fid, uint32_t ioctl_ctl_code,
                 const uint8_t *sndData, size_t sndDataLen,
                 uint8_t *rcvdData, size_t *rcvDataLen);
int smb2io_ntcreatex(void *smbctx, const char *path, const char *streamName,
                     struct open_inparms *inparms, 
                     struct open_outparm_ex *outparms, SMBFID *fid);
int smb2io_query_dir(void *smbctx, uint8_t file_info_class, uint8_t flags,
                     uint32_t file_index, SMBFID fid,
                     const char *name, uint32_t name_len,
                     char *rcv_output_buffer, uint32_t rcv_max_output_len,
                     uint32_t *rcv_output_len, uint32_t *query_dir_reply_len);
int smb2io_read(struct smb_ctx *smbctx, SMBFID fid, off_t offset, uint32_t count,
                char *dst, uint32_t *bytes_read);
int smb2io_transact(struct smb_ctx *smbctx, uint64_t *setup, int setupCnt, 
                    const char *pipeName, 
                    const uint8_t *sndPData, size_t sndPDataLen, 
                    const uint8_t *sndData, size_t sndDataLen, 
                    uint8_t *rcvPData, size_t *rcvPDataLen, 
                    uint8_t *rcvdData, size_t *rcvDataLen);
int smb2io_write(struct smb_ctx *smbctx, SMBFID fid, off_t offset, uint32_t count,
                 const char *src, uint32_t *bytes_written);

#endif
