/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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
#ifndef _VSOCK_H_
#define _VSOCK_H_

#include <sys/cdefs.h>

#include <sys/_types/_sa_family_t.h>
#include <sys/ucred.h>
#include <sys/socketvar.h>

__BEGIN_DECLS

#define VMADDR_CID_ANY (-1U)
#define VMADDR_CID_HYPERVISOR 0
#define VMADDR_CID_RESERVED 1
#define VMADDR_CID_HOST 2

#define VMADDR_PORT_ANY (-1U)

#define IOCTL_VM_SOCKETS_GET_LOCAL_CID _IOR('s',  209, uint32_t)

struct sockaddr_vm {
	__uint8_t      svm_len;        /* total length */
	sa_family_t    svm_family;     /* Address family: AF_VSOCK */
	__uint16_t     svm_reserved1;
	__uint32_t     svm_port;       /* Port # in host byte order */
	__uint32_t     svm_cid;        /* Address in host byte order */
} __attribute__((__packed__));

typedef u_quad_t vsock_gen_t;

struct xvsockpcb {
	u_int32_t      xv_len;            /* length of this structure */
	u_int64_t      xv_vsockpp;
	u_int32_t      xvp_local_cid;     /* local address cid */
	u_int32_t      xvp_local_port;    /* local address port */
	u_int32_t      xvp_remote_cid;    /* remote address cid */
	u_int32_t      xvp_remote_port;   /* remote address port */
	u_int32_t      xvp_rxcnt;         /* bytes received */
	u_int32_t      xvp_txcnt;         /* bytes transmitted */
	u_int32_t      xvp_peer_rxhiwat;  /* peer's receive buffer */
	u_int32_t      xvp_peer_rxcnt;    /* bytes received by peer */
	pid_t          xvp_last_pid;      /* last pid */
	vsock_gen_t    xvp_gencnt;        /* vsock generation count */
	struct xsocket xv_socket;
};

struct  xvsockpgen {
	u_int32_t      xvg_len;      /* length of this structure */
	u_int64_t      xvg_count;    /* number of PCBs at this time */
	vsock_gen_t    xvg_gen;      /* generation count at this time */
	so_gen_t       xvg_sogen;    /* current socket generation count */
};

__END_DECLS

#endif /* _VSOCK_H_ */
