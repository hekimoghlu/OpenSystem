/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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
#ifndef _NETSMB_SMB_RAP_H_
#define _NETSMB_SMB_RAP_H_

struct smb_rap {
	char *		r_sparam;
	char *		r_nparam;
	char *		r_sdata;
	char *		r_ndata;
	char *		r_pbuf;		/* rq parameters */
	int		r_plen;		/* rq param len */
	char *		r_npbuf;
	char *		r_dbuf;		/* rq data */
	int		r_dlen;		/* rq data len */
	char *		r_ndbuf;
	u_int32_t	r_result;
	char *		r_rcvbuf;
	int		r_rcvbuflen;
	int		r_entries;
};

struct smb_share_info_1 {
	char		shi1_netname[13];
	char		shi1_pad;
	u_int16_t	shi1_type;
	u_int32_t	shi1_remark;		/* char * */
};

__BEGIN_DECLS

int  smb_rap_create(int, const char *, const char *, struct smb_rap **);
void smb_rap_done(struct smb_rap *);
int  smb_rap_request(struct smb_rap *, struct smb_ctx *);
int  smb_rap_setNparam(struct smb_rap *, long);
int  smb_rap_setPparam(struct smb_rap *, void *);
int  smb_rap_error(struct smb_rap *, int);

int  smb_rap_NetShareEnum(struct smb_ctx *, int, void *, int, int *, int *);

__END_DECLS

#endif /* _NETSMB_SMB_RAP_H_ */
