/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
**      gssauthcn.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  The gssauth CN authentication module interface.
**
**
*/

#ifndef _GSSAUTHCN_H
#define _GSSAUTHCN_H 	1

#include <cn.h>

#if HAVE_GSS_FRAMEWORK
#include <GSS/gssapi.h>
#include <GSS/gssapi_spi.h>
#include <GSS/gssapi_krb5.h>
#else
#if HAVE_GSSAPI_GSSAPI_H
#include <gssapi/gssapi.h>
#endif
#if HAVE_GSSAPI_GSSAPI_KRB5_H
#include <gssapi/gssapi_krb5.h>
#endif
#if HAVE_GSSAPI_GSSAPI_EXT_H
#include <gssapi/gssapi_ext.h>
#endif
#endif

#if HAVE_KERBEROS_FRAMEWORK
#if TARGET_OS_EMBEDDED || TARGET_OS_IPHONE
#include <Heimdal/krb5.h>
#else
#include <Kerberos/krb5.h>
#endif
#else
#if HAVE_KRB5_H
#include <krb5.h>
#endif
#endif

typedef struct
{
    rpc_cn_auth_info_t cn_info;
    gss_ctx_id_t gss_ctx;
    OM_uint32 gss_stat;
    gss_OID gss_mech;
    boolean header_sign;
} rpc_gssauth_cn_info_t, *rpc_gssauth_cn_info_p_t;

PRIVATE rpc_protocol_id_t rpc__gssauth_negotiate_cn_init (
         rpc_auth_rpc_prot_epv_p_t      * /*epv*/,
         unsigned32                     * /*st*/
    );

PRIVATE rpc_protocol_id_t rpc__gssauth_mskrb_cn_init (
         rpc_auth_rpc_prot_epv_p_t      * /*epv*/,
         unsigned32                     * /*st*/
    );

PRIVATE rpc_protocol_id_t rpc__gssauth_winnt_cn_init (
         rpc_auth_rpc_prot_epv_p_t      * /*epv*/,
         unsigned32                     * /*st*/
    );

PRIVATE rpc_protocol_id_t rpc__gssauth_netlogon_cn_init (
         rpc_auth_rpc_prot_epv_p_t      * /*epv*/,
         unsigned32                     * /*st*/
    );

PRIVATE const char *rpc__gssauth_error_map (
	OM_uint32		/*major_status*/,
	OM_uint32		/*minor_status*/,
	const gss_OID		/*mech*/,
	char			* /*message_buffer*/,
	unsigned32		/*message_length*/,
	unsigned32		* /*st*/
    );

#endif /* _GSSAUTHCN_H */
