/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
_gsskrb5_pseudo_random(OM_uint32 *minor_status,
		       gss_ctx_id_t context_handle,
		       int prf_key,
		       const gss_buffer_t prf_in,
		       ssize_t desired_output_len,
		       gss_buffer_t prf_out)
{
    gsskrb5_ctx ctx = (gsskrb5_ctx)context_handle;
    krb5_context context;
    krb5_error_code ret;
    krb5_crypto crypto;
    krb5_data input, output;
    uint32_t num;
    OM_uint32 junk;
    unsigned char *p;
    krb5_keyblock *key = NULL;
    size_t dol;

    if (ctx == NULL) {
	*minor_status = 0;
	return GSS_S_NO_CONTEXT;
    }

    if (desired_output_len <= 0 || prf_in->length + 4 < prf_in->length) {
	*minor_status = 0;
	return GSS_S_FAILURE;
    }
    dol = desired_output_len;

    GSSAPI_KRB5_INIT (&context);

    switch(prf_key) {
    case GSS_C_PRF_KEY_FULL:
	_gsskrb5i_get_acceptor_subkey(ctx, context, &key);
	break;
    case GSS_C_PRF_KEY_PARTIAL:
	_gsskrb5i_get_initiator_subkey(ctx, context, &key);
	break;
    default:
	_gsskrb5_set_status(EINVAL, "unknown kerberos prf_key");
	*minor_status = EINVAL;
	return GSS_S_FAILURE;
    }

    if (key == NULL) {
	_gsskrb5_set_status(EINVAL, "no prf_key found");
	*minor_status = EINVAL;
	return GSS_S_FAILURE;
    }

    ret = krb5_crypto_init(context, key, 0, &crypto);
    krb5_free_keyblock (context, key);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }

    prf_out->value = malloc(dol);
    if (prf_out->value == NULL) {
	_gsskrb5_set_status(GSS_KRB5_S_KG_INPUT_TOO_LONG, "Out of memory");
	*minor_status = GSS_KRB5_S_KG_INPUT_TOO_LONG;
	krb5_crypto_destroy(context, crypto);
	return GSS_S_FAILURE;
    }
    prf_out->length = dol;

    HEIMDAL_MUTEX_lock(&ctx->ctx_id_mutex);

    input.length = prf_in->length + 4;
    input.data = malloc(prf_in->length + 4);
    if (input.data == NULL) {
	_gsskrb5_set_status(GSS_KRB5_S_KG_INPUT_TOO_LONG, "Out of memory");
	*minor_status = GSS_KRB5_S_KG_INPUT_TOO_LONG;
	gss_release_buffer(&junk, prf_out);
	krb5_crypto_destroy(context, crypto);
	HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
	return GSS_S_FAILURE;
    }
    memcpy(((uint8_t *)input.data) + 4, prf_in->value, prf_in->length);

    num = 0;
    p = prf_out->value;

    while(dol > 0) {
	size_t tsize;

	_gss_mg_encode_be_uint32(num, input.data);

	ret = krb5_crypto_prf(context, crypto, &input, &output);
	if (ret) {
	    *minor_status = ret;
	    free(input.data);
	    gss_release_buffer(&junk, prf_out);
	    krb5_crypto_destroy(context, crypto);
	    HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
	    return GSS_S_FAILURE;
	}

	tsize = min(dol, output.length);
	memcpy(p, output.data, tsize);
	p += tsize;
	dol -= tsize;
	krb5_data_free(&output);
	num++;
    }
    free(input.data);

    krb5_crypto_destroy(context, crypto);

    HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);

    return GSS_S_COMPLETE;
}
