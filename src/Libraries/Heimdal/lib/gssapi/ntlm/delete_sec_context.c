/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#include "ntlm.h"

OM_uint32 _gss_ntlm_delete_sec_context
           (OM_uint32 * minor_status,
            gss_ctx_id_t * context_handle,
            gss_buffer_t output_token
           )
{
    if (context_handle) {
	ntlm_ctx ctx = (ntlm_ctx)*context_handle;
	gss_cred_id_t cred = (gss_cred_id_t)ctx->client;
	size_t i;

	*context_handle = GSS_C_NO_CONTEXT;

	if (ctx->targetinfo.data != NULL)
	    free(ctx->targetinfo.data);

	if (ctx->ti.servername)
	    heim_ntlm_free_targetinfo(&ctx->ti);

	for (i = 0; i < ctx->num_backends; i++) {
	    if (ctx->backends[i].ctx == NULL)
		continue;
	    ctx->backends[i].interface->nsi_destroy(minor_status,
						    ctx->backends[i].ctx);
	}
	if (ctx->backends)
	    free(ctx->backends);

	if (ctx->srcname)
	    _gss_ntlm_release_name(NULL, &ctx->srcname);
	if (ctx->targetname)
	    _gss_ntlm_release_name(NULL, &ctx->targetname);
	if (ctx->clientsuppliedtargetname)
	    free(ctx->clientsuppliedtargetname);

	
	_gss_ntlm_destroy_crypto(ctx);

	krb5_data_free(&ctx->sessionkey);
	krb5_data_free(&ctx->type1);
	krb5_data_free(&ctx->type2);
	krb5_data_free(&ctx->type3);
	gss_release_buffer(minor_status, &ctx->pac);

	_gss_ntlm_release_cred(NULL, &cred);

	memset(ctx, 0, sizeof(*ctx));
	free(ctx);
    }
    if (output_token) {
	output_token->length = 0;
	output_token->value  = NULL;
    }

    *minor_status = 0;
    return GSS_S_COMPLETE;
}
