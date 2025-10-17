/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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

#if 0
OM_uint32
gss_krb5_copy_ccache(OM_uint32 *minor_status,
		     krb5_context context,
		     gss_cred_id_t cred,
		     krb5_ccache out)
{
    krb5_error_code kret;

    HEIMDAL_MUTEX_lock(&cred->cred_id_mutex);

    if (cred->ccache == NULL) {
	HEIMDAL_MUTEX_unlock(&cred->cred_id_mutex);
	*minor_status = EINVAL;
	return GSS_S_FAILURE;
    }

    kret = krb5_cc_copy_cache(context, cred->ccache, out);
    HEIMDAL_MUTEX_unlock(&cred->cred_id_mutex);
    if (kret) {
	*minor_status = kret;
	return GSS_S_FAILURE;
    }
    *minor_status = 0;
    return GSS_S_COMPLETE;
}
#endif


OM_uint32
_gsskrb5_krb5_import_cred(OM_uint32 *minor_status,
			  krb5_ccache id,
			  krb5_principal keytab_principal,
			  krb5_keytab keytab,
			  gss_cred_id_t *cred)
{
    krb5_context context;
    krb5_error_code kret;
    gsskrb5_cred handle;
    OM_uint32 ret;

    *cred = NULL;

    GSSAPI_KRB5_INIT (&context);

    handle = calloc(1, sizeof(*handle));
    if (handle == NULL) {
	_gsskrb5_clear_status ();
	*minor_status = ENOMEM;
        return (GSS_S_FAILURE);
    }
    HEIMDAL_MUTEX_init(&handle->cred_id_mutex);

    handle->usage = 0;

    if (id) {
	char *str;

	handle->usage |= GSS_C_INITIATE;

	kret = krb5_cc_get_principal(context, id,
				     &handle->principal);
	if (kret) {
	    free(handle);
	    *minor_status = kret;
	    return GSS_S_FAILURE;
	}

	if (keytab_principal) {
	    krb5_boolean match;

	    match = krb5_principal_compare(context,
					   handle->principal,
					   keytab_principal);
	    if (match == FALSE) {
		krb5_free_principal(context, handle->principal);
		free(handle);
		_gsskrb5_clear_status ();
		*minor_status = EINVAL;
		return GSS_S_FAILURE;
	    }
	}

	ret = __gsskrb5_ccache_lifetime(minor_status,
					context,
					id,
					handle->principal,
					&handle->endtime);
	if (ret != GSS_S_COMPLETE) {
	    krb5_free_principal(context, handle->principal);
	    free(handle);
	    return ret;
	}


	kret = krb5_cc_get_full_name(context, id, &str);
	if (kret)
	    goto out;

	kret = krb5_cc_resolve(context, str, &handle->ccache);
	free(str);
	if (kret)
	    goto out;
    }


    if (keytab) {
	char *str;

	handle->usage |= GSS_C_ACCEPT;

	if (keytab_principal && handle->principal == NULL) {
	    kret = krb5_copy_principal(context,
				       keytab_principal,
				       &handle->principal);
	    if (kret)
		goto out;
	}

	kret = krb5_kt_get_full_name(context, keytab, &str);
	if (kret)
	    goto out;

	kret = krb5_kt_resolve(context, str, &handle->keytab);
	free(str);
	if (kret)
	    goto out;
    }


    *minor_status = 0;
    *cred = (gss_cred_id_t)handle;
    return GSS_S_COMPLETE;

out:
    if (handle->ccache)
	krb5_cc_close(context, handle->ccache);
    if (handle->keytab)
	krb5_kt_close(context, handle->keytab);
    if (handle->principal)
	krb5_free_principal(context, handle->principal);
    HEIMDAL_MUTEX_destroy(&handle->cred_id_mutex);
    free(handle);
    *minor_status = kret;
    return GSS_S_FAILURE;
}
