/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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
gss_export_sec_context(OM_uint32 *__nonnull minor_status,
    __nullable gss_ctx_id_t * __nonnull context_handle,
    __nullable gss_buffer_t interprocess_token)
{
	OM_uint32 major_status;
	struct _gss_context *ctx;
	gssapi_mech_interface m;
	gss_buffer_desc buf;

	*minor_status = 0;

	if (interprocess_token)
	    _mg_buffer_zero(interprocess_token);
	else
	    return GSS_S_CALL_INACCESSIBLE_READ;

	if (context_handle == NULL)
	    return GSS_S_NO_CONTEXT;

	ctx = (struct _gss_context *) *context_handle;
	if (ctx == NULL || ctx->gc_ctx == NULL) {
	    *minor_status = 0;
	    return GSS_S_NO_CONTEXT;
	}

	m = ctx->gc_mech;

	major_status = m->gm_export_sec_context(minor_status,
	    &ctx->gc_ctx, &buf);

	if (major_status == GSS_S_COMPLETE) {
		unsigned char *p;

		free(ctx);
		*context_handle = GSS_C_NO_CONTEXT;
		interprocess_token->length = buf.length
			+ 2 + m->gm_mech_oid.length;
		interprocess_token->value = malloc(interprocess_token->length);
		if (!interprocess_token->value) {
			/*
			 * We are in trouble here - the context is
			 * already gone. This is allowed as long as we
			 * set the caller's context_handle to
			 * GSS_C_NO_CONTEXT, which we did above.
			 * Return GSS_S_FAILURE.
			 */
			_mg_buffer_zero(interprocess_token);
			*minor_status = ENOMEM;
			return (GSS_S_FAILURE);
		}
		p = interprocess_token->value;
		p[0] = m->gm_mech_oid.length >> 8;
		p[1] = m->gm_mech_oid.length;
		memcpy(p + 2, m->gm_mech_oid.elements, m->gm_mech_oid.length);
		memcpy(p + 2 + m->gm_mech_oid.length, buf.value, buf.length);
		gss_release_buffer(minor_status, &buf);
	} else {
		_gss_mg_error(m, *minor_status);
	}

	return (major_status);
}
