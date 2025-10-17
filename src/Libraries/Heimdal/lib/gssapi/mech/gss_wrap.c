/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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

/**
 * Wrap a message using either confidentiality (encryption +
 * signature) or sealing (signature).
 *
 * @param minor_status minor status code.
 * @param context_handle context handle.
 * @param conf_req_flag if non zero, confidentiality is requestd.
 * @param qop_req type of protection needed, in most cases it GSS_C_QOP_DEFAULT should be passed in.
 * @param input_message_buffer messages to wrap
 * @param conf_state returns non zero if confidentiality was honoured.
 * @param output_message_buffer the resulting buffer, release with gss_release_buffer().
 *
 * @ingroup gssapi
 */

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_wrap(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_ctx_id_t context_handle,
    int conf_req_flag,
    gss_qop_t qop_req,
    __nonnull const gss_buffer_t input_message_buffer,
    int *__nullable conf_state,
    __nonnull gss_buffer_t output_message_buffer)
{
	struct _gss_context *ctx = (struct _gss_context *) context_handle;
	gssapi_mech_interface m;

	if (conf_state)
	    *conf_state = 0;
	_mg_buffer_zero(output_message_buffer);
	if (ctx == NULL) {
	    *minor_status = 0;
	    return GSS_S_NO_CONTEXT;
	}

	m = ctx->gc_mech;

	return (m->gm_wrap(minor_status, ctx->gc_ctx,
		    conf_req_flag, qop_req, input_message_buffer,
		    conf_state, output_message_buffer));
}
