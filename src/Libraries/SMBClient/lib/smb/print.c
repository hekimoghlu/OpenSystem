/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
#include <sys/param.h>
#include <sys/sysctl.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/mount.h>
#include <fcntl.h>
#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pwd.h>
#include <grp.h>
#include <unistd.h>

#include <netsmb/upi_mbuf.h>
#include <sys/mchain.h>
#include <netsmb/smb_lib.h>
#include <netsmb/rq.h>
#include <netsmb/smb_conn.h>
#include <netsmb/smb_converter.h>

int
smb_smb_open_print_file(struct smb_ctx *ctx, int setuplen, int mode,
	const char *ident, smbfh *fhp)
{
	struct smb_usr_rq *rqp;
	mbchain_t mbp;
	int error;

	error = smb_usr_rq_init(ctx, SMB_COM_OPEN_PRINT_FILE, 0, &rqp);
	if (error)
		return error;
	mbp = smb_usr_rq_getrequest(rqp);
	smb_usr_rq_wstart(rqp);
	mb_put_uint16le(mbp, setuplen);
	mb_put_uint16le(mbp, mode);
	smb_usr_rq_wend(rqp);
	smb_usr_rq_bstart(rqp);
	mb_put_uint8(mbp, SMB_DT_ASCII);
	smb_usr_rq_put_dstring(ctx, mbp, ident, strlen(ident), SMB_UTF_SFM_CONVERSIONS, NULL);
	smb_usr_rq_bend(rqp);
	error = smb_usr_rq_simple(rqp);
	if (!error) {
		mdchain_t mdp;

		mdp = smb_usr_rq_getreply(rqp);
		md_get_uint8(mdp, NULL);	/* Word Count */
		md_get_uint16(mdp, fhp);
	}
	smb_usr_rq_done(rqp);
	return error;
}

int
smb_smb_close_print_file(struct smb_ctx *ctx, smbfh fh)
{
	struct smb_usr_rq *rqp;
	mbchain_t mbp;
	int error;

	error = smb_usr_rq_init(ctx, SMB_COM_CLOSE_PRINT_FILE, 0, &rqp);
	if (error)
		return error;
	mbp = smb_usr_rq_getrequest(rqp);
	smb_usr_rq_wstart(rqp);
	mb_put_mem(mbp, (char*)&fh, 2, MB_MSYSTEM);
	smb_usr_rq_wend(rqp);
	smb_usr_rq_bstart(rqp);
	smb_usr_rq_bend(rqp);
	error = smb_usr_rq_simple(rqp);
	smb_usr_rq_done(rqp);
	return error;
}
