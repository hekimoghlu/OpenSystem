/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
**  NAME
**
**      schnauthcn.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  The netlogon/schannel CN authentication module interface.
**
**
*/

#ifndef _SCHNAUTHCN_H
#define _SCHNAUTHCN_H	1

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <cn.h>

#include <schannel.h>

typedef struct rpc_schnauth_cn_info
{
    rpc_cn_auth_info_t  cn_info;

    /*
     * Schannel security context
     */

    struct schn_auth_ctx sec_ctx;

} rpc_schnauth_cn_info_t, *rpc_schnauth_cn_info_p_t;

typedef struct rpc_schnauth_creds
{
    unsigned32 flags1;
    unsigned32 flags2;
    unsigned_char_p_t domain_name;
    unsigned_char_p_t machine_name;
} rpc_schnauth_creds_t, *rpc_schnauth_creds_p_t;

typedef struct rpc_cn_schnauth_tlr
{
    unsigned8 signature[8];
    unsigned8 seq_number[8];
    unsigned8 digest[8];
    unsigned8 nonce[8];

} rpc_cn_schnauth_tlr_t, *rpc_cn_schnauth_tlr_p_t;

#define RPC_CN_PKT_SIZEOF_SCHNAUTH_TLR  32

EXTERNAL rpc_cn_auth_epv_t rpc_g_schnauth_cn_epv;

#endif /* _SCHNAUTHCN_H */
