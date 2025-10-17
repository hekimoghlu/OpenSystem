/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
gss_delete_sec_context(OM_uint32 * __nonnull minor_status,
    __nullable gss_ctx_id_t * __nonnull context_handle,
    __nullable gss_buffer_t output_token)
{
	OM_uint32 major_status, junk;
	struct _gss_context *ctx = (struct _gss_context *) *context_handle;

	if (output_token)
	    _mg_buffer_zero(output_token);

	*minor_status = 0;
	major_status = GSS_S_COMPLETE;
    
	if (ctx) {
		if (ctx->gc_replaced_cred)
			gss_release_cred(&junk, &ctx->gc_replaced_cred);

		/*
		 * If we have an implementation ctx, delete it,
		 * otherwise fake an empty token.
		 */
		if (ctx->gc_ctx) {
			major_status = ctx->gc_mech->gm_delete_sec_context(
				minor_status, &ctx->gc_ctx, output_token);
		}
		free(ctx);
		*context_handle = GSS_C_NO_CONTEXT;
	}

	return major_status;
}
