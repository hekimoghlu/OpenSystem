/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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

OM_uint32
_netlogon_delete_sec_context(OM_uint32 * minor_status,
                             gss_ctx_id_t * context_handle,
                             gss_buffer_t output_token)
{
    if (context_handle != NULL && *context_handle != GSS_C_NO_CONTEXT) {
        gssnetlogon_ctx ctx = (gssnetlogon_ctx)*context_handle;

        *context_handle = GSS_C_NO_CONTEXT;

        _netlogon_release_name(minor_status, (gss_name_t *)&ctx->SourceName);
        _netlogon_release_name(minor_status, (gss_name_t *)&ctx->TargetName);
        HEIMDAL_MUTEX_destroy(&ctx->Mutex);
        memset(ctx, 0, sizeof(*ctx));
        free(ctx);
    }

    if (output_token != GSS_C_NO_BUFFER) {
        output_token->length = 0;
        output_token->value  = NULL;
    }

    *minor_status = 0;
    return GSS_S_COMPLETE;
}
