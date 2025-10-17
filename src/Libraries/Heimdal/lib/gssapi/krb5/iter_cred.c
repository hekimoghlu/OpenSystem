/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
#include <heim_threads.h>

static void
iter_creds_f(OM_uint32 flags,
	     gss_OID type,
	     void *userctx ,
	     void (*cred_iter)(void *, gss_OID, gss_cred_id_t))
{
    krb5_context context;
    krb5_cccol_cursor cursor;
    krb5_error_code ret;
    krb5_ccache id;

    GSSAPI_KRB5_INIT_GOTO(&context, out);

    ret = krb5_cccol_cursor_new (context, &cursor);
    if (ret)
	goto out;

    while (krb5_cccol_cursor_next (context, cursor, &id) == 0 && id != NULL) {
	gsskrb5_cred handle;
	OM_uint32 junk;
	krb5_principal principal;
	krb5_data data;
	gss_OID resolved_type = NULL;

	ret = krb5_cc_get_principal(context, id, &principal);
	if (ret) {
	    krb5_cc_close(context, id);
	    continue;
	}
	
	if (krb5_principal_is_pku2u(context, principal))
	    resolved_type = GSS_PKU2U_MECHANISM;
	else if (krb5_cc_get_config(context, id, NULL, "iakerb", &data) == 0) {
	    resolved_type = GSS_IAKERB_MECHANISM;
	    krb5_data_free(&data);
	} else {
	    resolved_type = GSS_KRB5_MECHANISM;
	}

	if (!gss_oid_equal(type, resolved_type)) {
	    krb5_free_principal(context, principal);
	    krb5_cc_close(context, id);
	    continue;
	}

	handle = calloc(1, sizeof(*handle));
	if (handle == NULL) {
	    krb5_cc_close(context, id);
	    goto out;
	}

	HEIMDAL_MUTEX_init(&handle->cred_id_mutex);
	
	handle->usage = GSS_C_INITIATE;
	handle->principal = principal;

	__gsskrb5_ccache_lifetime(&junk, context, id,
				  handle->principal, &handle->endtime);
	handle->keytab = NULL;
	handle->ccache = id;

	cred_iter(userctx, type, (gss_cred_id_t)handle);
    }

    krb5_cccol_cursor_free(context, &cursor);

 out:
    cred_iter(userctx, NULL, NULL);
}		 

void
_gss_pku2u_iter_creds_f(OM_uint32 flags,
			void *userctx ,
			void (*cred_iter)(void *, gss_OID, gss_cred_id_t))
{
    iter_creds_f(flags, GSS_PKU2U_MECHANISM, userctx, cred_iter);
}

void
_gss_krb5_iter_creds_f(OM_uint32 flags,
		       void *userctx ,
		       void (*cred_iter)(void *, gss_OID, gss_cred_id_t))
{
    iter_creds_f(flags, GSS_KRB5_MECHANISM, userctx, cred_iter);
}

void
_gss_iakerb_iter_creds_f(OM_uint32 flags,
		       void *userctx ,
		       void (*cred_iter)(void *, gss_OID, gss_cred_id_t))
{
    iter_creds_f(flags, GSS_IAKERB_MECHANISM, userctx, cred_iter);
}
