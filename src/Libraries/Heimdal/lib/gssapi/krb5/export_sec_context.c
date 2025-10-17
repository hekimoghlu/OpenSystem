/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
_gsskrb5_export_sec_context (
    OM_uint32 * minor_status,
    gss_ctx_id_t * context_handle,
    gss_buffer_t interprocess_token
    )
{
    krb5_context context;
    const gsskrb5_ctx ctx = (const gsskrb5_ctx) *context_handle;
    krb5_storage *sp;
    krb5_auth_context ac;
    OM_uint32 ret = GSS_S_COMPLETE;
    krb5_data data;
    gss_buffer_desc buffer;
    int flags;
    OM_uint32 minor;
    krb5_error_code kret;

    GSSAPI_KRB5_INIT (&context);

    HEIMDAL_MUTEX_lock(&ctx->ctx_id_mutex);

    if (!(ctx->flags & GSS_C_TRANS_FLAG)) {
	HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
	*minor_status = 0;
	return GSS_S_UNAVAILABLE;
    }

    sp = krb5_storage_emem ();
    if (sp == NULL) {
	HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }
    ac = ctx->auth_context;

    /* flagging included fields */

    flags = 0;
    if (ac->local_address)
	flags |= SC_LOCAL_ADDRESS;
    if (ac->remote_address)
	flags |= SC_REMOTE_ADDRESS;
    if (ac->keyblock)
	flags |= SC_KEYBLOCK;
    if (ac->local_subkey)
	flags |= SC_LOCAL_SUBKEY;
    if (ac->remote_subkey)
	flags |= SC_REMOTE_SUBKEY;

    kret = krb5_store_int32 (sp, flags);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }

    /* marshall auth context */

    kret = krb5_store_int32 (sp, ac->flags);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }
    if (ac->local_address) {
	kret = krb5_store_address (sp, *ac->local_address);
	if (kret) {
	    *minor_status = kret;
	    goto failure;
	}
    }
    if (ac->remote_address) {
	kret = krb5_store_address (sp, *ac->remote_address);
	if (kret) {
	    *minor_status = kret;
	    goto failure;
	}
    }
    kret = krb5_store_int16 (sp, ac->local_port);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }
    kret = krb5_store_int16 (sp, ac->remote_port);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }
    if (ac->keyblock) {
	kret = krb5_store_keyblock (sp, *ac->keyblock);
	if (kret) {
	    *minor_status = kret;
	    goto failure;
	}
    }
    if (ac->local_subkey) {
	kret = krb5_store_keyblock (sp, *ac->local_subkey);
	if (kret) {
	    *minor_status = kret;
	    goto failure;
	}
    }
    if (ac->remote_subkey) {
	kret = krb5_store_keyblock (sp, *ac->remote_subkey);
	if (kret) {
	    *minor_status = kret;
	    goto failure;
	}
    }
    kret = krb5_store_int32 (sp, ac->local_seqnumber);
	if (kret) {
	    *minor_status = kret;
	    goto failure;
	}
    kret = krb5_store_int32 (sp, ac->remote_seqnumber);
	if (kret) {
	    *minor_status = kret;
	    goto failure;
	}

    kret = krb5_store_int32 (sp, ac->keytype);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }
    kret = krb5_store_int32 (sp, ac->cksumtype);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }

    /* names */

    ret = _gsskrb5_export_name (minor_status,
				(gss_name_t)ctx->source, &buffer);
    if (ret)
	goto failure;
    data.data   = buffer.value;
    data.length = buffer.length;
    kret = krb5_store_data (sp, data);
    _gsskrb5_release_buffer (&minor, &buffer);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }

    ret = _gsskrb5_export_name (minor_status,
				(gss_name_t)ctx->target, &buffer);
    if (ret)
	goto failure;
    data.data   = buffer.value;
    data.length = buffer.length;

    ret = GSS_S_FAILURE;

    kret = krb5_store_data (sp, data);
    _gsskrb5_release_buffer (&minor, &buffer);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }

    kret = krb5_store_int32 (sp, ctx->flags);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }
    kret = krb5_store_int32 (sp, ctx->more_flags);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }
    kret = krb5_store_int32 (sp, (int32_t)ctx->endtime);
    if (kret) {
	*minor_status = kret;
	goto failure;
    }
    kret = _gssapi_msg_order_export(sp, ctx->gk5c.order);
    if (kret ) {
        *minor_status = kret;
        goto failure;
    }

    kret = krb5_storage_to_data (sp, &data);
    krb5_storage_free (sp);
    if (kret) {
	HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
	*minor_status = kret;
	return GSS_S_FAILURE;
    }
    interprocess_token->length = data.length;
    interprocess_token->value  = data.data;
    HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
    ret = _gsskrb5_delete_sec_context (minor_status, context_handle,
				       GSS_C_NO_BUFFER);
    if (ret != GSS_S_COMPLETE)
	_gsskrb5_release_buffer (NULL, interprocess_token);
    *minor_status = 0;
    return ret;
 failure:
    HEIMDAL_MUTEX_unlock(&ctx->ctx_id_mutex);
    krb5_storage_free (sp);
    return ret;
}
