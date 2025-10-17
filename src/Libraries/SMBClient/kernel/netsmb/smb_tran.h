/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#ifndef _NETSMB_SMB_TRAN_H_
#define	_NETSMB_SMB_TRAN_H_

#include <sys/socket.h>

/*
 * Known transports
 */
#define	SMBT_NBTCP	1

/*
 * Transport parameters
 */
#define	SMBTP_SNDSZ     1   /* R  - int */
#define	SMBTP_RCVSZ     2   /* R  - int */
#define	SMBTP_TIMEOUT   3   /* RW - struct timespec */
#define	SMBTP_SELECTID  4   /* RW - (void *) */
#define SMBTP_UPCALL    5   /* RW - (* void)(void *) */
#define SMBTP_QOS       6   /* RW - uint32_t */
#define SMBTP_IP_ADDR   7   /* R  - struct sockaddr_storage */
#define SMBTP_BOUND_IF  8   /* W  - uint32_t */

struct smb_tran_ops;

struct smb_tran_desc {
	sa_family_t	tr_type;
	int	 (*tr_create)(struct smbiod *iod);                          /* smb_nbst_create */
	int	 (*tr_done)(struct smbiod *iod);                            /* smb_nbst_done */
	int	 (*tr_bind)(struct smbiod *iod, struct sockaddr *sap);      /* smb_nbst_bind */
	int	 (*tr_connect)(struct smbiod *iod, struct sockaddr *sap);   /* smb_nbst_connect */
	int	 (*tr_disconnect)(struct smbiod *iod);                      /* smb_nbst_disconnect */
	int	 (*tr_send)(struct smbiod *iod, mbuf_t m0);                 /* smb_nbst_send */
	int	 (*tr_recv)(struct smbiod *iod, mbuf_t *mpp);               /* smb_nbst_recv */
	void (*tr_timo)(struct smbiod *iod);                            /* smb_nbst_timo */
	int	 (*tr_getparam)(struct smbiod *iod, int param, void *data); /* smb_nbst_getparam */
	int	 (*tr_setparam)(struct smbiod *iod, int param, void *data); /* smb_nbst_setparam */
	int	 (*tr_fatal)(struct smbiod *iod, int error);                /* smb_nbst_fatal */
	LIST_ENTRY(smb_tran_desc)	tr_link;
};

#define SMB_TRAN_CREATE(iod)            (iod)->iod_tdesc->tr_create(iod)
#define SMB_TRAN_DONE(iod)              (iod)->iod_tdesc->tr_done(iod)
#define	SMB_TRAN_BIND(iod,sap)          (iod)->iod_tdesc->tr_bind(iod,sap)
#define	SMB_TRAN_CONNECT(iod,sap)       (iod)->iod_tdesc->tr_connect(iod,sap)
#define	SMB_TRAN_DISCONNECT(iod)        (iod)->iod_tdesc->tr_disconnect(iod)
#define	SMB_TRAN_SEND(iod,m0)           (iod)->iod_tdesc->tr_send(iod,m0)
#define	SMB_TRAN_RECV(iod,m)            (iod)->iod_tdesc->tr_recv(iod,m)
#define	SMB_TRAN_TIMO(iod)              (iod)->iod_tdesc->tr_timo(iod)
#define	SMB_TRAN_GETPARAM(iod,par,data) (iod)->iod_tdesc->tr_getparam(iod, par, data)
#define	SMB_TRAN_SETPARAM(iod,par,data) (iod)->iod_tdesc->tr_setparam(iod, par, data)
#define	SMB_TRAN_FATAL(iod, error)      (iod)->iod_tdesc->tr_fatal(iod, error)

#endif /* _NETSMB_SMB_TRAN_H_ */
