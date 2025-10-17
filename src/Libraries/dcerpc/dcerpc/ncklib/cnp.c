/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
**      cnp.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Definitions of global variables.
**
**
*/

#include <commonp.h>    /* Common declarations for all RPC runtime */
#include <com.h>        /* Common communications services */
#include <comprot.h>    /* Common protocol services */
#include <cnp.h>        /* NCA Connection private declarations */

GLOBAL rpc_cond_t               rpc_g_cn_lookaside_cond;
GLOBAL rpc_list_desc_t          rpc_g_cn_syntax_lookaside_list;
GLOBAL rpc_list_desc_t          rpc_g_cn_sec_lookaside_list;
GLOBAL rpc_list_desc_t          rpc_g_cn_assoc_lookaside_list;
GLOBAL rpc_list_desc_t          rpc_g_cn_binding_lookaside_list;
GLOBAL rpc_list_desc_t          rpc_g_cn_lg_fbuf_lookaside_list;
GLOBAL rpc_list_desc_t          rpc_g_cn_sm_fbuf_lookaside_list;
GLOBAL rpc_list_desc_t          rpc_g_cn_call_lookaside_list;
GLOBAL rpc_cn_assoc_grp_tbl_t   rpc_g_cn_assoc_grp_tbl;
GLOBAL unsigned32               rpc_g_cn_call_id;
GLOBAL rpc_cn_mgmt_t            rpc_g_cn_mgmt;
