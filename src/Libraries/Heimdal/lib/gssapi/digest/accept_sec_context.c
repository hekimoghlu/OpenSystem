/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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

/*
 * Allocate a scram context handle for the first provider that
 * is up and running.
 */
OM_uint32
_gss_scram_allocate_ctx(OM_uint32 *minor_status, const char *domain, scram_id_t *ctx)
{
    scram_id_t c;

    *ctx = NULL;

    c = calloc(1, sizeof(*c));
    if (c == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    *ctx = c;

    return GSS_S_COMPLETE;

}

/*
 *
 */

OM_uint32
_gss_scram_accept_sec_context
(OM_uint32 * minor_status,
 gss_ctx_id_t * context_handle,
 const gss_cred_id_t acceptor_cred_handle,
 const gss_buffer_t input_token_buffer,
 const gss_channel_bindings_t input_chan_bindings,
 gss_name_t * src_name,
 gss_OID * mech_type,
 gss_buffer_t output_token,
 OM_uint32 * ret_flags,
 OM_uint32 * time_rec,
 gss_cred_id_t * delegated_cred_handle
    )
{
    *minor_status = 0;

    if (context_handle == NULL)
	return GSS_S_FAILURE;
	
    if (src_name)
	*src_name = GSS_C_NO_NAME;
    if (mech_type)
	*mech_type = GSS_C_NO_OID;
    if (ret_flags)
	*ret_flags = 0;
    if (time_rec)
	*time_rec = 0;
    if (delegated_cred_handle)
	*delegated_cred_handle = GSS_C_NO_CREDENTIAL;

#if 0
    if (*context_handle == GSS_C_NO_CONTEXT) {
	OM_uint32 major_status;
	OM_uint32 retflags = 0;

	_gss_mg_log(10, "scram-asc-s1");

	major_status = _gss_scram_allocate_ctx(minor_status, NULL, &ctx);
	if (major_status)
	    return major_status;
	*context_handle = (gss_ctx_id_t)ctx;
	
	ctx->flags = retflags;

	return GSS_S_CONTINUE_NEEDED;
    } else {
	OM_uint32 maj_stat;
	size_t i;

	if (input_token_buffer == GSS_C_NO_BUFFER)
	    return GSS_S_FAILURE;

	ctx = (scram_id_t)*context_handle;

	data.data = input_token_buffer->value;
	data.length = input_token_buffer->length;

	ctx->client = strdup("lha");


	_gss_mg_log(10, "scram-asc-s2");

	if (src_name)
	    *src_name = (gss_name_t)strdup(ctx->client);

	if (mech_type)
	    *mech_type = GSS_SCRAM_MECHANISM;
	if (time_rec)
	    *time_rec = GSS_C_INDEFINITE;

	ctx->status |= STATUS_OPEN;

	if (ret_flags)
	    *ret_flags = ctx->flags;

	return GSS_S_FAILURE;
    }
#else
    return GSS_S_FAILURE;
#endif
}
