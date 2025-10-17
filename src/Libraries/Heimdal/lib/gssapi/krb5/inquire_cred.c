/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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

OM_uint32 GSSAPI_CALLCONV _gsskrb5_inquire_cred
(OM_uint32 * minor_status,
 const gss_cred_id_t cred_handle,
 gss_name_t * output_name,
 OM_uint32 * lifetime,
 gss_cred_usage_t * cred_usage,
 gss_OID_set * mechanisms
    )
{
    krb5_context context;
    gss_cred_id_t aqcred_init = GSS_C_NO_CREDENTIAL;
    gss_cred_id_t aqcred_accept = GSS_C_NO_CREDENTIAL;
    gsskrb5_cred acred = NULL, icred = NULL;
    OM_uint32 ret;

    *minor_status = 0;

    if (output_name)
	*output_name = NULL;
    if (mechanisms)
	*mechanisms = GSS_C_NO_OID_SET;

    GSSAPI_KRB5_INIT (&context);

    if (cred_handle == GSS_C_NO_CREDENTIAL) {
	ret = _gsskrb5_acquire_cred(minor_status,
				    GSS_C_NO_NAME,
				    GSS_C_INDEFINITE,
				    GSS_C_NO_OID_SET,
				    GSS_C_ACCEPT,
				    &aqcred_accept,
				    NULL,
				    NULL);
	if (ret == GSS_S_COMPLETE)
	    acred = (gsskrb5_cred)aqcred_accept;

	ret = _gsskrb5_acquire_cred(minor_status,
				    GSS_C_NO_NAME,
				    GSS_C_INDEFINITE,
				    GSS_C_NO_OID_SET,
				    GSS_C_INITIATE,
				    &aqcred_init,
				    NULL,
				    NULL);
	if (ret == GSS_S_COMPLETE)
	    icred = (gsskrb5_cred)aqcred_init;

	if (icred == NULL && acred == NULL) {
	    *minor_status = 0;
	    return GSS_S_NO_CRED;
	}
    } else {
	gsskrb5_cred handle;
	
	handle = (gsskrb5_cred)cred_handle;

	if (handle->keytab)
	    acred = handle;
	else
	    icred = handle;

	/*
	 * double check that the credential is still in the cache if
	 * we have a initiator only cache.
	 */

	if (handle->usage == GSS_C_INITIATE && handle->principal != NULL) {
	    krb5_error_code kret;
	    krb5_ccache ccache;

	    kret = krb5_cc_cache_match(context, handle->principal, &ccache);
	    if (kret == 0) {
		krb5_cc_close(context, ccache);
	    } else {
		*minor_status = kret;
		return GSS_S_DEFECTIVE_CREDENTIAL;
	    }
	}
    }

    if (acred)
	HEIMDAL_MUTEX_lock(&acred->cred_id_mutex);
    if (icred)
	HEIMDAL_MUTEX_lock(&icred->cred_id_mutex);

    if (output_name != NULL) {
	if (icred && icred->principal != NULL) {
	    gss_name_t name;

	    if (acred && acred->principal)
		name = (gss_name_t)acred->principal;
	    else
		name = (gss_name_t)icred->principal;

            ret = _gsskrb5_duplicate_name(minor_status, name, output_name);
            if (ret)
		goto out;
	} else if (acred && acred->usage == GSS_C_ACCEPT) {
	    krb5_principal princ;
	    *minor_status = krb5_sname_to_principal(context, NULL,
						    NULL, KRB5_NT_SRV_HST,
						    &princ);
	    if (*minor_status) {
		ret = GSS_S_FAILURE;
		goto out;
	    }
	    *output_name = (gss_name_t)princ;
	} else {
	    krb5_principal princ;
	    *minor_status = krb5_get_default_principal(context,
						       &princ);
	    if (*minor_status) {
		ret = GSS_S_FAILURE;
		goto out;
	    }
	    *output_name = (gss_name_t)princ;
	}
    }
    if (lifetime != NULL) {
	time_t aendtime = INT_MAX, iendtime = INT_MAX;

	if (acred) aendtime = acred->endtime;
	if (icred) iendtime = icred->endtime;

	ret = _gsskrb5_lifetime_left(minor_status,
				     context,
				     min(aendtime, iendtime),
				     lifetime);
	if (ret)
	    goto out;
    }
    if (cred_usage != NULL) {
	if (acred && icred)
	    *cred_usage = GSS_C_BOTH;
	else if (acred)
	    *cred_usage = GSS_C_ACCEPT;
	else if (icred)
	    *cred_usage = GSS_C_INITIATE;
	else
	    abort();
    }

    if (mechanisms != NULL) {
        ret = gss_create_empty_oid_set(minor_status, mechanisms);
        if (ret)
	    goto out;
	ret = gss_add_oid_set_member(minor_status, GSS_KRB5_MECHANISM,
				     mechanisms);
        if (ret)
	    goto out;
    }
    ret = GSS_S_COMPLETE;
out:
    if (acred)
	HEIMDAL_MUTEX_unlock(&acred->cred_id_mutex);
    if (icred)
	HEIMDAL_MUTEX_unlock(&icred->cred_id_mutex);

    if (aqcred_init != GSS_C_NO_CREDENTIAL)
	ret = _gsskrb5_release_cred(minor_status, &aqcred_init);
    if (aqcred_accept != GSS_C_NO_CREDENTIAL)
	ret = _gsskrb5_release_cred(minor_status, &aqcred_accept);

    return ret;
}
