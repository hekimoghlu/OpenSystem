/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
#include "netlogon.h"

OM_uint32 _netlogon_inquire_context (
            OM_uint32 * minor_status,
            const gss_ctx_id_t context_handle,
            gss_name_t * src_name,
            gss_name_t * targ_name,
            OM_uint32 * lifetime_rec,
            gss_OID * mech_type,
            OM_uint32 * ctx_flags,
            int * locally_initiated,
            int * open_context
           )
{
    const gssnetlogon_ctx ctx = (const gssnetlogon_ctx)context_handle;
    OM_uint32 ret;

    if (src_name != NULL) {
        ret = _netlogon_duplicate_name(minor_status, (gss_name_t)ctx->SourceName,
                                       (gss_name_t *)src_name);
        if (GSS_ERROR(ret))
            return ret;
    }
    if (targ_name != NULL) {
        ret = _netlogon_duplicate_name(minor_status, (gss_name_t)ctx->TargetName,
                                       (gss_name_t *)targ_name);
        if (GSS_ERROR(ret))
            return ret;
    }
    if (mech_type != NULL)
        *mech_type = GSS_NETLOGON_MECHANISM;
    if (ctx_flags != NULL)
        *ctx_flags = ctx->GssFlags;
    if (locally_initiated != NULL)
        *locally_initiated = ctx->LocallyInitiated;
    if (open_context != NULL)
        *open_context = (ctx->State == NL_AUTH_ESTABLISHED);

    return GSS_S_COMPLETE;
}

