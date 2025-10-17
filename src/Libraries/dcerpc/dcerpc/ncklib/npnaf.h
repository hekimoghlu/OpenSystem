/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
/*
**
**  NAME
**
**      npnaf.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Definitions and Data Type declarations
**  used by Windows NT Named Pipes Network Address Family Extension
**  service.
**
**
*/

#ifndef _NPNAF_H
#define _NPNAF_H	1

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef sec_id_base_v0_0_included
#include <dce/id_base.h>
#endif

/***********************************************************************
 *
 *  Include the Internet specific socket address
 */

#ifdef HAVE_SYS_UN_H
#include <sys/un.h>
#else
#include <un.h>
#endif

#ifndef RPC_C_NP_DIR
#define RPC_C_NP_DIR	"/var/opt/novell/xad/rpc"
#endif

#ifndef RPC_C_NP_DIR_LEN
#define RPC_C_NP_DIR_LEN (sizeof(RPC_C_NP_DIR) - 1)
#endif

#define RPC_C_NP_SEC_CONTEXT_MIN_LEN    (4 /*Length*/ + 4 /*Version*/ + 4/*UserNameLength*/ + 4/*DomainNameLength*/ + 4/*SessionKeyLength*/)
#define RPC_C_NP_SEC_CONTEXT_MAX_LEN	(1024)

/* NetBIOS name length */
#define RPC_C_NETADDR_NP_MAX            18

/* Rest is for the path. */
#define RPC_C_ENDPOINT_NP_MAX           (RPC_C_PATH_NP_MAX - RPC_C_NETADDR_NP_MAX - 1)

/***********************************************************************
 *
 *  The representation of an RPC Address that holds an NP address.
 */

typedef struct rpc_addr_np_t
{
    rpc_protseq_id_t        rpc_protseq_id;
    socklen_t               len;
    struct sockaddr_un      sa;
    char                    remote_host[PATH_MAX];
} rpc_np_addr_t, *rpc_np_addr_p_t;

/*
 * Max Local DG Fragment Size:
 *
 *   The size in bytes of the largest DG fragment that can be sent to
 *   a "local" address. The data won't be transmitted over the "wire"
 *   by the transport service, i.e., the loopback is done on the local
 *   host. This is determined when the socket is created and won't
 *   change in the life of the socket.
 *
 * The constant defined here is based on experimentation.
 *
 * Caution: This must be less than RPC_C_DG_MAX_FRAG_SIZE::dg.h!
 */

#ifndef RPC_C_NP_MAX_LOCAL_FRAG_SIZE
#define RPC_C_NP_MAX_LOCAL_FRAG_SIZE (8 * 1024)
#endif

/***********************************************************************
 *
 *  Routine Prototypes for the Internet Extension service routines.
 */

#ifdef __cplusplus
extern "C" {
#endif

PRIVATE void rpc__np_init (
        rpc_naf_epv_p_t             * /*naf_epv*/,
        unsigned32                  * /*status*/
    );

PRIVATE void rpc__np_desc_inq_addr (
        rpc_protseq_id_t             /*protseq_id*/,
        rpc_socket_t                 /*desc*/,
        rpc_addr_vector_p_t         * /*rpc_addr_vec*/,
        unsigned32                  * /*st*/
    );

PRIVATE void rpc__np_get_broadcast (
        rpc_naf_id_t                 /*naf_id*/,
        rpc_protseq_id_t             /*rpc_protseq_id*/,
        rpc_addr_vector_p_t         * /*rpc_addrs*/,
        unsigned32                  * /*status*/
    );

PRIVATE void rpc__np_init_local_addr_vec (
        unsigned32                  * /*status*/
    );

#if 0
    /* Removed unused symbol for rdar://problem/26430747 */
PRIVATE boolean32 rpc__np_is_local_network (
        rpc_addr_p_t                 /*rpc_addr*/,
        unsigned32                  * /*status*/
    );

PRIVATE boolean32 rpc__np_is_local_addr (
        rpc_addr_p_t                 /*rpc_addr*/,
        unsigned32                  * /*status*/
    );
#endif

PRIVATE boolean32 rpc__np_is_valid_endpoint (
	const unsigned_char_t      * /* endpoint */,
	unsigned32                 * /* status */
    );

PRIVATE void rpc__np_naf_init_func(void);

#ifdef __cplusplus
}
#endif

#endif /* _NPNAF_H */
