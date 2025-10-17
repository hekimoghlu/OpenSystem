/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#ifndef __NETSMB_SMBIO_H_INCLUDED__
#define __NETSMB_SMBIO_H_INCLUDED__

#include <stdint.h>
#include <sys/types.h>
#include <netsmb/smb_lib.h>

struct open_inparms {
	uint32_t rights;
	uint32_t shareMode;
	uint32_t disp;
	uint32_t attrs;
	uint64_t allocSize;
	uint32_t createOptions;
};

struct open_outparm {
	uint64_t createTime;
	uint64_t accessTime;
	uint64_t writeTime;
	uint64_t changeTime;
	uint32_t attributes;
	uint64_t allocationSize;
	uint64_t fileSize;
	uint8_t volumeGID[16];
	uint64_t fileInode;
	uint32_t maxAccessRights;
	uint32_t maxGuessAccessRights;
};

/*  Return value is -errno if < 0, otherwise the received byte count. */
ssize_t smbio_read(void *smbctx, int fid, uint8_t *buf, size_t bufSize);

/* 
 * Perform a smb transaction call
 *
 * Return zero if no error or the appropriate errno.
 */
int smbio_transact(void *smbctx, uint16_t *setup, int setupCnt, const char *name, 
				   const uint8_t *sndPData, size_t sndPDataLen, 
				   const uint8_t *sndData, size_t sndDataLen, 
				   uint8_t *rcvPData, size_t *rcvPDataLen, 
				   uint8_t *rcvdData, size_t *rcvDataLen);

/* Open a SMB names pipe. Returns 0 on success, otherwise -errno. */
int smbio_open_pipe(void * smbctx, const char *	pipename, int *fid);
int smbio_close_file(void *ctx, int fid);
int smbio_check_directory(struct smb_ctx *ctx, const void *path, 
						  uint32_t /* flags2 */, uint32_t */* nt_error */);
int smbio_ntcreatex(void *smbctx, const char *path, const char *streamName, 
					struct open_inparms *inparms, struct open_outparm *outparms, 
					int *fid);
#endif /* __NETSMB_SMBIO_H_INCLUDED__ */

