/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
gss_import_sec_context(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_buffer_t interprocess_token,
    __nullable gss_ctx_id_t * __nonnull context_handle)
{
	OM_uint32 major_status;
	gssapi_mech_interface m;
	struct _gss_context *ctx;
	gss_OID_desc mech_oid;
	gss_buffer_desc buf;
	unsigned char *p;
	size_t len;

	*minor_status = 0;
	*context_handle = GSS_C_NO_CONTEXT;

	/*
	 * We added an oid to the front of the token in
	 * gss_export_sec_context.
	 */
	p = interprocess_token->value;
	len = interprocess_token->length;
	if (len < 2)
		return (GSS_S_DEFECTIVE_TOKEN);
	mech_oid.length = (p[0] << 8) | p[1];
	if (len < mech_oid.length + 2)
		return (GSS_S_DEFECTIVE_TOKEN);
	mech_oid.elements = p + 2;
	buf.length = len - 2 - mech_oid.length;
	buf.value = p + 2 + mech_oid.length;

	m = __gss_get_mechanism(&mech_oid);
	if (!m)
		return (GSS_S_DEFECTIVE_TOKEN);

	ctx = malloc(sizeof(struct _gss_context));
	if (!ctx) {
		*minor_status = ENOMEM;
		return (GSS_S_FAILURE);
	}
	memset(ctx, 0, sizeof(struct _gss_context));
	ctx->gc_mech = m;
	major_status = m->gm_import_sec_context(minor_status,
	    &buf, &ctx->gc_ctx);
	if (major_status != GSS_S_COMPLETE) {
		_gss_mg_error(m, *minor_status);
		free(ctx);
	} else {
		*context_handle = (gss_ctx_id_t) ctx;
	}

	return (major_status);
}
