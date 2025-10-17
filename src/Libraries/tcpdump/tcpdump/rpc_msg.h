/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
 * rpc_msg.h
 * rpc message definition
 *
 * Copyright (C) 1984, Sun Microsystems, Inc.
 */

#define SUNRPC_MSG_VERSION	((uint32_t) 2)

/*
 * Bottom up definition of an rpc message.
 * NOTE: call and reply use the same overall struct but
 * different parts of unions within it.
 */

enum sunrpc_msg_type {
	SUNRPC_CALL=0,
	SUNRPC_REPLY=1
};

enum sunrpc_reply_stat {
	SUNRPC_MSG_ACCEPTED=0,
	SUNRPC_MSG_DENIED=1
};

enum sunrpc_accept_stat {
	SUNRPC_SUCCESS=0,
	SUNRPC_PROG_UNAVAIL=1,
	SUNRPC_PROG_MISMATCH=2,
	SUNRPC_PROC_UNAVAIL=3,
	SUNRPC_GARBAGE_ARGS=4,
	SUNRPC_SYSTEM_ERR=5
};

enum sunrpc_reject_stat {
	SUNRPC_RPC_MISMATCH=0,
	SUNRPC_AUTH_ERROR=1
};

/*
 * Reply part of an rpc exchange
 */

/*
 * Reply to an rpc request that was rejected by the server.
 */
struct sunrpc_rejected_reply {
	nd_uint32_t		 rj_stat;	/* enum reject_stat */
	union {
		struct {
			nd_uint32_t low;
			nd_uint32_t high;
		} RJ_versions;
		nd_uint32_t RJ_why;  /* enum auth_stat - why authentication did not work */
	} ru;
#define	rj_vers	ru.RJ_versions
#define	rj_why	ru.RJ_why
};

/*
 * Body of a reply to an rpc request.
 */
struct sunrpc_reply_body {
	nd_uint32_t	rp_stat;		/* enum reply_stat */
	struct sunrpc_rejected_reply rp_reject;	/* if rejected */
};

/*
 * Body of an rpc request call.
 */
struct sunrpc_call_body {
	nd_uint32_t cb_rpcvers;	/* must be equal to two */
	nd_uint32_t cb_prog;
	nd_uint32_t cb_vers;
	nd_uint32_t cb_proc;
	struct sunrpc_opaque_auth cb_cred;
	/* followed by opaque verifier */
};

/*
 * The rpc message
 */
struct sunrpc_msg {
	nd_uint32_t		rm_xid;
	nd_uint32_t		rm_direction;	/* enum msg_type */
	union {
		struct sunrpc_call_body RM_cmb;
		struct sunrpc_reply_body RM_rmb;
	} ru;
#define	rm_call		ru.RM_cmb
#define	rm_reply	ru.RM_rmb
};
#define	acpted_rply	ru.RM_rmb.ru.RP_ar
#define	rjcted_rply	ru.RM_rmb.ru.RP_dr
