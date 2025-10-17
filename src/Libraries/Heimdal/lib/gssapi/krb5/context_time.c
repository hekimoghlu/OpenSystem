/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

OM_uint32
_gsskrb5_lifetime_left(OM_uint32 *minor_status,
		       krb5_context context,
		       time_t endtime,
		       OM_uint32 *lifetime)
{
    krb5_timestamp timeret;
    krb5_error_code kret;

    if (endtime == INT_MAX) {
	*lifetime = GSS_C_INDEFINITE;
	return GSS_S_COMPLETE;
    }

    kret = krb5_timeofday(context, &timeret);
    if (kret) {
	*minor_status = kret;
	return GSS_S_FAILURE;
    }

    if (endtime < timeret)
	*lifetime = 0;
    else
	*lifetime = (OM_uint32)(endtime - timeret);

    return GSS_S_COMPLETE;
}


OM_uint32 GSSAPI_CALLCONV _gsskrb5_context_time
           (OM_uint32 * minor_status,
            const gss_ctx_id_t context_handle,
            OM_uint32 * time_rec
           )
{
    krb5_context context;
    time_t endtime;
    OM_uint32 major_status;
    const gsskrb5_ctx ctx = (const gsskrb5_ctx) context_handle;

    GSSAPI_KRB5_INIT (&context);

    HEIMDAL_MUTEX_lock(&ctx->ctx_id_mutex);
    endtime = ctx->endtime;
    HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);

    major_status = _gsskrb5_lifetime_left(minor_status, context,
					  endtime, time_rec);
    if (major_status != GSS_S_COMPLETE)
	return major_status;

    *minor_status = 0;

    if (*time_rec == 0)
	return GSS_S_CONTEXT_EXPIRED;

    return GSS_S_COMPLETE;
}
