/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#include "gssdigest.h"

OM_uint32 _gss_scram_inquire_context (
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
    scram_id_t ctx = (scram_id_t)context_handle;

    *minor_status = 0;

    if (ctx == NULL)
	return GSS_S_CALL_BAD_STRUCTURE;

    if (src_name) {
	if (ctx->client == NULL)
	    return GSS_S_NO_CONTEXT;
#ifdef HAVE_KCM
	*src_name = (gss_name_t)strdup(ctx->client);
#else
	*src_name = (gss_name_t)strdup(ctx->client->name);
#endif
    }
    if (targ_name)
	*targ_name = GSS_C_NO_NAME;
    if (lifetime_rec)
	*lifetime_rec = GSS_C_INDEFINITE;
    if (mech_type)
	*mech_type = GSS_SCRAM_MECHANISM;
    if (ctx_flags)
	*ctx_flags = ctx->flags;
    if (locally_initiated)
	*locally_initiated = (ctx->status & STATUS_CLIENT) ? 1 : 0;
    if (open_context)
	*open_context = (ctx->status & STATUS_OPEN) ? 1 : 0;

    return GSS_S_COMPLETE;
}
