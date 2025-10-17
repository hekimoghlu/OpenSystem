/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
**  NAME:
**
**      dgglob.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  DG-specific runtime global variable (external) declarations.
**
**
*/

#ifndef _DGGLOB_H
#define _DGGLOB_H

/* ========================================================================= */

/*
 * Client connection table (CCT).
 */
EXTERNAL rpc_dg_cct_t rpc_g_dg_cct;

/*
 * Client call control block table (CCALLT).
 */
EXTERNAL rpc_dg_ccall_p_t rpc_g_dg_ccallt[];

/*
 * Server connection table (SCT)
 */
EXTERNAL rpc_dg_sct_elt_p_t rpc_g_dg_sct[];

#ifndef NO_STATS

/*
 * RPC statistics
 */
EXTERNAL rpc_dg_stats_t rpc_g_dg_stats;

#endif /* NO_STATS */

/*
 * The server boot time
 */
EXTERNAL unsigned32 rpc_g_dg_server_boot_time;

/*
 * The following UUID will be used to uniquely identify a particular instance
 * of a client process.  It is periodically sent to all servers with which we
 * need to maintain liveness.
 */

EXTERNAL idl_uuid_t rpc_g_dg_my_cas_uuid;

/*
 * The DG socket pool
 */
EXTERNAL rpc_dg_sock_pool_t rpc_g_dg_sock_pool;

#endif /* _DGGLOB_H */
