/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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
#include <netsmb/smb_lib.h>
#include <netsmb/smb_conn.h>

int
smb_read(struct smb_ctx *ctx, smbfh fh, off_t offset, uint32_t count, char *dst)
{
	struct smbioc_rw rwrq;

	bzero(&rwrq, sizeof(rwrq));
	rwrq.ioc_version = SMB_IOC_STRUCT_VERSION;
	rwrq.ioc_fh = fh;
	rwrq.ioc_base = dst;
	rwrq.ioc_cnt = count;
	rwrq.ioc_offset = offset;
	if (smb_ioctl_call(ctx->ct_fd, SMBIOC_READ, &rwrq) == -1) {
		return -1;
	}
	return rwrq.ioc_cnt;
}

int 
smb_write(struct smb_ctx *ctx, smbfh fh, off_t offset, uint32_t count, const char *src)
{
	struct smbioc_rw rwrq;

	bzero(&rwrq, sizeof(rwrq));
	rwrq.ioc_version = SMB_IOC_STRUCT_VERSION;
	rwrq.ioc_fh = fh;
	rwrq.ioc_base = (char *)src;
	rwrq.ioc_cnt = count;
	rwrq.ioc_offset = offset;
	/* 
	 * Curretly we don't support Write Modes from user land. We do support paasing
	 * it down, but until we see a requirement lets leave it zero out.
	 */
	if (smb_ioctl_call(ctx->ct_fd, SMBIOC_WRITE, &rwrq) == -1) {
		return -1;
	}
	return rwrq.ioc_cnt;
}
