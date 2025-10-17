/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
_gsskrb5_store_cred(OM_uint32         *minor_status,
		    gss_cred_id_t     input_cred_handle,
		    gss_cred_usage_t  cred_usage,
		    const gss_OID     desired_mech,
		    OM_uint32         overwrite_cred,
		    OM_uint32         default_cred,
		    gss_OID_set       *elements_stored,
		    gss_cred_usage_t  *cred_usage_stored)
{
    krb5_context context;
    krb5_error_code ret;
    gsskrb5_cred cred;
    krb5_ccache id;
    int destroy = 0;

    *minor_status = 0;

    if (cred_usage != GSS_C_INITIATE) {
	*minor_status = GSS_KRB5_S_G_BAD_USAGE;
	return GSS_S_FAILURE;
    }

    if (gss_oid_equal(desired_mech, GSS_KRB5_MECHANISM) == 0)
	return GSS_S_BAD_MECH;

    cred = (gsskrb5_cred)input_cred_handle;
    if (cred == NULL)
	return GSS_S_NO_CRED;

    GSSAPI_KRB5_INIT (&context);

    HEIMDAL_MUTEX_lock(&cred->cred_id_mutex);
    if (cred->usage != cred_usage && cred->usage != GSS_C_BOTH) {
	HEIMDAL_MUTEX_unlock(&cred->cred_id_mutex);
	*minor_status = GSS_KRB5_S_G_BAD_USAGE;
	return(GSS_S_FAILURE);
    }

    if (cred->principal == NULL) {
	HEIMDAL_MUTEX_unlock(&cred->cred_id_mutex);
	*minor_status = GSS_KRB5_S_KG_TGT_MISSING;
	return(GSS_S_FAILURE);
    }

    /* write out cred to credential cache */

    ret = krb5_cc_cache_match(context, cred->principal, &id);
    if (ret) {
	ret = krb5_cc_new_unique(context, NULL, NULL, &id);
	if (ret) {
	    HEIMDAL_MUTEX_unlock(&cred->cred_id_mutex);
	    *minor_status = ret;
	    return(GSS_S_FAILURE);
	}
	destroy = 1;
    }

    ret = krb5_cc_initialize(context, id, cred->principal);
    if (ret == 0)
	ret = krb5_cc_copy_match_f(context, cred->ccache, id, NULL, NULL, NULL);
    if (ret) {
	if (destroy)
	    krb5_cc_destroy(context, id);
	else
	    krb5_cc_close(context, id);
	HEIMDAL_MUTEX_unlock(&cred->cred_id_mutex);
	*minor_status = ret;
	return(GSS_S_FAILURE);
    }

    if (default_cred)
	krb5_cc_switch(context, id);

    krb5_cc_close(context, id);

    HEIMDAL_MUTEX_unlock(&cred->cred_id_mutex);

    *minor_status = 0;
    return GSS_S_COMPLETE;
}
