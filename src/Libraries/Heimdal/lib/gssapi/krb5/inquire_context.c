/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
#include "gsskrb5_locl.h"

OM_uint32 GSSAPI_CALLCONV _gsskrb5_inquire_context (
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
    krb5_context context;
    OM_uint32 ret;
    gsskrb5_ctx ctx = (gsskrb5_ctx)context_handle;

    if (src_name)
	*src_name = GSS_C_NO_NAME;
    if (targ_name)
	*targ_name = GSS_C_NO_NAME;

    GSSAPI_KRB5_INIT (&context);

    HEIMDAL_MUTEX_lock(&ctx->ctx_id_mutex);

    if (src_name) {
	if (ctx->source == NULL) {
	    ret = KRB5KDC_ERR_CLIENT_NOTYET;
	    goto failed;
	}
	ret = _gsskrb5_duplicate_name(minor_status,
				      (gss_name_t)ctx->source,
				      src_name);
	if (ret)
	    goto failed;
    }

    if (targ_name) {
	if (ctx->target == NULL) {
	    ret = KRB5KDC_ERR_SERVICE_NOTYET;
	    goto failed;
	}
	ret = _gsskrb5_duplicate_name(minor_status,
				      (gss_name_t)ctx->target,
				      targ_name);
	if (ret)
	    goto failed;
    }

    if (lifetime_rec) {
	ret = _gsskrb5_lifetime_left(minor_status,
				     context,
				     ctx->endtime,
				     lifetime_rec);
	if (ret)
	    goto failed;
    }

    if (mech_type)
	*mech_type = ctx->mech;

    if (ctx_flags)
	*ctx_flags = ctx->flags;

    if (locally_initiated)
	*locally_initiated = ctx->more_flags & LOCAL;

    if (open_context)
	*open_context = ctx->more_flags & OPEN;

    *minor_status = 0;

    HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
    return GSS_S_COMPLETE;

failed:
    if (src_name)
	_gsskrb5_release_name(NULL, src_name);
    if (targ_name)
	_gsskrb5_release_name(NULL, targ_name);

    HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
    return ret;
}
