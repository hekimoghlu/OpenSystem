/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#ifndef _NETSMB_RQ_H_
#define _NETSMB_RQ_H_

#include <sys/types.h>

struct smb_usr_rq;

int smb_ntwrkpath_to_localpath(struct smb_ctx *, 
							   const char */*ntwrkstr*/, size_t /*ntwrk_len*/,
							   char */*utf8str*/, size_t */*utf8_len*/,
							   uint32_t /*flags*/);
int smb_localpath_to_ntwrkpath(struct smb_ctx *,
							   const char */*utf8str*/, size_t /*utf8_len*/,
							   char */*ntwrkstr*/, size_t */*ntwrk_len*/,
							   uint32_t /*flags*/);

/*
 * Calling smb_usr_rq_init_rcvsize with a request size causes it to allocate a 
 * receive buffer of that size. 
 */
int  smb_usr_rq_init_rcvsize(struct smb_ctx *, u_char, uint16_t, size_t, struct smb_usr_rq **);
/* The smb_usr_rq_init routtine will always allocate a receive buffer of page size. */
int  smb_usr_rq_init(struct smb_ctx *, u_char, uint16_t, struct smb_usr_rq **);
void smb_usr_rq_done(struct smb_usr_rq *);
mbchain_t smb_usr_rq_getrequest(struct smb_usr_rq *);
mdchain_t smb_usr_rq_getreply(struct smb_usr_rq *);
uint32_t smb_usr_rq_get_error(struct smb_usr_rq *);
uint32_t smb_usr_rq_flags2(struct smb_usr_rq *);
uint32_t smb_usr_rq_nt_error(struct smb_usr_rq *rqp);
void smb_usr_rq_setflags2(struct smb_usr_rq *, uint32_t );
void smb_usr_rq_wstart(struct smb_usr_rq *);
void smb_usr_rq_wend(struct smb_usr_rq *);
void smb_usr_rq_bstart(struct smb_usr_rq *);
void smb_usr_rq_bend(struct smb_usr_rq *);
int smb_usr_rq_simple(struct smb_usr_rq *);
int smb_usr_put_dmem(struct smb_ctx *, mbchain_t , const char *, 
            			size_t , int /*flags*/, size_t *);
int smb_usr_rq_put_dstring(struct smb_ctx *, mbchain_t , const char *, size_t, 
							int /*flags*/, size_t *);

int smb_usr_t2_request(struct smb_ctx *ctx, int setupcount, uint16_t *setup, const char *name, 
				   		uint16_t tparamcnt, const void *tparam, 
				   		uint16_t tdatacnt, const void *tdata, 
				   		uint16_t *rparamcnt, void *rparam, 
				   		uint16_t *rdatacnt, void *rdata, 
				   		uint32_t *buffer_oflow);
#endif // _NETSMB_RQ_H_
