/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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
**      dgglob.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Defining instances of DG-specific runtime global variables.
**
**
*/

#include <dg.h>

/* ======= */

    /*
     * The following variables are the "defining instance" and static
     * initialization of variables mentioned (and documented) in
     * "dgglob.h".
     */

GLOBAL rpc_dg_cct_t rpc_g_dg_cct;

GLOBAL rpc_dg_ccall_p_t rpc_g_dg_ccallt[RPC_DG_CCALLT_SIZE];

GLOBAL rpc_dg_sct_elt_p_t rpc_g_dg_sct[RPC_DG_SCT_SIZE];

GLOBAL rpc_dg_stats_t rpc_g_dg_stats = RPC_DG_STATS_INITIALIZER;

GLOBAL unsigned32 rpc_g_dg_server_boot_time;

GLOBAL idl_uuid_t rpc_g_dg_my_cas_uuid;

GLOBAL rpc_dg_sock_pool_t rpc_g_dg_sock_pool;
/* ======= */
