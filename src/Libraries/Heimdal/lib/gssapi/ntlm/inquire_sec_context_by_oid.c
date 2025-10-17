/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#include <gssapi_krb5.h>

OM_uint32
_gss_ntlm_inquire_sec_context_by_oid(OM_uint32 *minor_status,
				     const gss_ctx_id_t context_handle,
				     const gss_OID desired_object,
				     gss_buffer_set_t *data_set)
{
    ntlm_ctx ctx = (ntlm_ctx)context_handle;

    if (ctx == NULL) {
	*minor_status = 0;
	return gss_mg_set_error_string(GSS_NTLM_MECHANISM, GSS_S_NO_CONTEXT,
				       0, "no context");
    }

    *minor_status = 0;
    *data_set = GSS_C_NO_BUFFER_SET;

    if (gss_oid_equal(desired_object, GSS_NTLM_GET_SESSION_KEY_X) ||
        gss_oid_equal(desired_object, GSS_C_INQ_SSPI_SESSION_KEY)) {
	gss_buffer_desc value;

	if (ctx->sessionkey.length == 0) {
	    *minor_status = ENOENT;
	    return GSS_S_FAILURE;
	}

	value.length = ctx->sessionkey.length;
	value.value = ctx->sessionkey.data;

	return gss_add_buffer_set_member(minor_status,
					 &value,
					 data_set);
    } else if (gss_oid_equal(desired_object, GSS_C_INQ_WIN2K_PAC_X)) {
	if (ctx->pac.length == 0) {
	    *minor_status = ENOENT;
	    return GSS_S_FAILURE;
	}

	return gss_add_buffer_set_member(minor_status,
					 &ctx->pac,
					 data_set);
    } else if (gss_oid_equal(desired_object, GSS_C_NTLM_GUEST)) {
	gss_buffer_desc value;
	uint32_t num;

	if (ctx->kcmflags & KCM_NTLM_FLAG_AV_GUEST)
	    num = 1;
	else
	    num = 0;

	value.length = sizeof(num);
	value.value = &num;

	return gss_add_buffer_set_member(minor_status,
					 &value,
					 data_set);
    } else if (gss_oid_equal(desired_object, GSS_C_PEER_HAS_UPDATED_SPNEGO)) {
	if (ctx->flags & NTLM_NEG_NTLM2_SESSION)
	    return GSS_S_COMPLETE;
	return GSS_S_FAILURE;
    } else if (gss_oid_equal(desired_object, GSS_C_NTLM_RESET_KEYS)) {
	_gss_ntlm_set_keys(ctx);
	return GSS_S_COMPLETE;
    } else {
	*minor_status = 0;
	return GSS_S_FAILURE;
    }
}
