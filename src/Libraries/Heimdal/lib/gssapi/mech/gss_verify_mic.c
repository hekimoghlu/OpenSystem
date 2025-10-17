/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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
gss_verify_mic(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_ctx_id_t context_handle,
    __nonnull const gss_buffer_t message_buffer,
    __nonnull const gss_buffer_t token_buffer,
    gss_qop_t * __nullable qop_state)
{
	struct _gss_context *ctx = (struct _gss_context *) context_handle;
	gssapi_mech_interface m;

	if (qop_state)
	    *qop_state = 0;
	if (ctx == NULL) {
	    *minor_status = 0;
	    return GSS_S_NO_CONTEXT;
	}

	m = ctx->gc_mech;

	return (m->gm_verify_mic(minor_status, ctx->gc_ctx,
		    message_buffer, token_buffer, qop_state));
}
