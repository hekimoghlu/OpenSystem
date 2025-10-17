/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#include "mech_locl.h"

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_inquire_context(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_ctx_id_t context_handle,
    __nullable gss_name_t * __nullable src_name,
    __nullable gss_name_t * __nullable targ_name,
    OM_uint32 * __nullable lifetime_rec,
    __nullable gss_OID * __nullable mech_type,
    OM_uint32 * __nullable ctx_flags,
    int * __nullable locally_initiated,
    int * __nullable xopen)
{
	OM_uint32 major_status;
	struct _gss_context *ctx = (struct _gss_context *) context_handle;
	gssapi_mech_interface m;
	struct _gss_name *name;
	gss_name_t src_mn, targ_mn;

	if (locally_initiated)
	    *locally_initiated = 0;
	if (xopen)
	    *xopen = 0;
	if (lifetime_rec)
	    *lifetime_rec = 0;

	if (src_name)
	    *src_name = GSS_C_NO_NAME;
	if (targ_name)
	    *targ_name = GSS_C_NO_NAME;
	if (mech_type)
	    *mech_type = GSS_C_NO_OID;
	src_mn = targ_mn = GSS_C_NO_NAME;

	if (ctx == NULL || ctx->gc_ctx == NULL) {
	    *minor_status = 0;
	    return GSS_S_NO_CONTEXT;
	}

	m = ctx->gc_mech;

	major_status = m->gm_inquire_context(minor_status,
	    ctx->gc_ctx,
	    src_name ? &src_mn : NULL,
	    targ_name ? &targ_mn : NULL,
	    lifetime_rec,
	    mech_type,
	    ctx_flags,
	    locally_initiated,
	    xopen);

	if (major_status != GSS_S_COMPLETE) {
		_gss_mg_error(m, *minor_status);
		return (major_status);
	}

	if (src_name) {
		name = _gss_create_name(src_mn, m);
		if (!name) {
			if (mech_type)
				*mech_type = GSS_C_NO_OID;
			m->gm_release_name(minor_status, &src_mn);
			*minor_status = 0;
			return (GSS_S_FAILURE);
		}
		*src_name = (gss_name_t) name;
	}

	if (targ_name) {
		name = _gss_create_name(targ_mn, m);
		if (!name) {
			if (mech_type)
				*mech_type = GSS_C_NO_OID;
			if (src_name)
				gss_release_name(minor_status, src_name);
			m->gm_release_name(minor_status, &targ_mn);
			*minor_status = 0;
			return (GSS_S_FAILURE);
		}
		*targ_name = (gss_name_t) name;
	}

	return (GSS_S_COMPLETE);
}
