/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

OM_uint32 GSSAPI_CALLCONV
_gsskrb5_delete_sec_context(OM_uint32 * minor_status,
			    gss_ctx_id_t * context_handle,
			    gss_buffer_t output_token)
{
    krb5_context context;
    gsskrb5_ctx ctx;

    GSSAPI_KRB5_INIT (&context);

    *minor_status = 0;

    if (output_token) {
	output_token->length = 0;
	output_token->value  = NULL;
    }

    if (*context_handle == GSS_C_NO_CONTEXT)
	return GSS_S_COMPLETE;

    ctx = (gsskrb5_ctx) *context_handle;
    *context_handle = GSS_C_NO_CONTEXT;

    HEIMDAL_MUTEX_lock(&ctx->ctx_id_mutex);

    if (ctx->ccache) {
	if (ctx->more_flags & CLOSE_CCACHE)
	    krb5_cc_close(context, ctx->ccache);
	else if (ctx->more_flags & DESTROY_CCACHE)
	    krb5_cc_destroy(context, ctx->ccache);
    }

    krb5_auth_con_free (context, ctx->auth_context);
    krb5_auth_con_free (context, ctx->deleg_auth_context);
    if (ctx->kcred)
	krb5_free_creds(context, ctx->kcred);
    if(ctx->source)
	krb5_free_principal (context, ctx->source);
    if(ctx->target)
	krb5_free_principal (context, ctx->target);
    if (ctx->ticket)
	krb5_free_ticket (context, ctx->ticket);
    if(ctx->gk5c.order)
	_gssapi_msg_order_destroy(&ctx->gk5c.order);
    if (ctx->service_keyblock)
	krb5_free_keyblock (context, ctx->service_keyblock);
    krb5_data_free(&ctx->fwd_data);
    if (ctx->gk5c.crypto)
    	krb5_crypto_destroy(context, ctx->gk5c.crypto);
    if (ctx->tgsctx) {
	krb5_tkt_creds_free(context, ctx->tgsctx);
    }
#ifdef PKINIT
    if (ctx->cert)
	hx509_cert_free(ctx->cert);
    if (ctx->gic_opt)
	krb5_get_init_creds_opt_free(context, ctx->gic_opt);
#endif
    if (ctx->asctx)
	krb5_init_creds_free(context, ctx->asctx);
    if (ctx->password) {
	memset(ctx->password, 0, strlen(ctx->password));
	free(ctx->password);
    }
    if (ctx->cookie)
	krb5_free_data(context, ctx->cookie);
    if (ctx->iakerbrealm)
	free(ctx->iakerbrealm);
    if (ctx->messages)
	krb5_storage_free(ctx->messages);
    if (ctx->friendlyname.length)
	krb5_data_free(&ctx->friendlyname);
    if (ctx->lkdchostname.length)
	krb5_data_free(&ctx->lkdchostname);

    HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
    HEIMDAL_MUTEX_destroy(&ctx->ctx_id_mutex);
    memset(ctx, 0, sizeof(*ctx));
    free (ctx);
    return GSS_S_COMPLETE;
}
