/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
**      bindcall.c
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Provides canned routines that may be used in conjunction with the
**      [binding_callout] ACF attribute.  These routines are called from a
**      client stub to possibly modify the binding handle, typically with
**      security information.
**
*/
#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <dce/rpc.h>
#include <dce/stubbase.h>

#if 0
/* Removed unused symbol for rdar://problem/26430747 */
void rpc_ss_bind_authn_client
(
    rpc_binding_handle_t    *p_bh,      /* [io] Binding handle */
    rpc_if_handle_t         if_h,       /* [in] Interface handle */
    error_status_t          *p_st       /*[out] Return status */
)
{
    unsigned_char_t *princ_name;        /* Server principal name */

    /* Resolve binding handle if not fully bound */
    rpc_ep_resolve_binding(*p_bh, if_h, p_st);
    if (*p_st != rpc_s_ok)
        return;

    /* Get server principal name */
    rpc_mgmt_inq_server_princ_name(
        *p_bh,                          /* binding handle */
        (unsigned32) rpc_c_authn_default, /* default authentication service */
        &princ_name,                    /* server principal name */
        p_st);
    if (*p_st != rpc_s_ok)
        return;

    /* Set auth info in binding handle */
    rpc_binding_set_auth_info(
        *p_bh,                          /* binding handle */
        princ_name,                     /* server principal name */
        (unsigned32) rpc_c_protect_level_default, /* default protection level */
        (unsigned32) rpc_c_authn_default, /* default authentication service */
        NULL,                           /* def. auth credentials (login ctx) */
        (unsigned32) rpc_c_authz_name,  /* authz based on cli principal name */
        p_st);
}
#endif
