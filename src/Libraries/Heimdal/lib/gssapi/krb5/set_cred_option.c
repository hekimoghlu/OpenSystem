/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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

static OM_uint32
import_cred(OM_uint32 *minor_status,
	    krb5_context context,
            gss_cred_id_t *cred_handle,
            const gss_buffer_t value)
{
    OM_uint32 major_stat;
    krb5_error_code ret;
    krb5_principal keytab_principal = NULL;
    krb5_keytab keytab = NULL;
    krb5_storage *sp = NULL;
    krb5_ccache id = NULL;
    char *str;

    if (cred_handle == NULL || *cred_handle != GSS_C_NO_CREDENTIAL) {
	*minor_status = 0;
	return GSS_S_FAILURE;
    }

    sp = krb5_storage_from_mem(value->value, value->length);
    if (sp == NULL) {
	*minor_status = 0;
	return GSS_S_FAILURE;
    }

    /* credential cache name */
    ret = krb5_ret_string(sp, &str);
    if (ret) {
	*minor_status = ret;
	major_stat =  GSS_S_FAILURE;
	goto out;
    }
    if (str[0]) {
	ret = krb5_cc_resolve(context, str, &id);
	if (ret) {
	    *minor_status = ret;
	    major_stat =  GSS_S_FAILURE;
	    goto out;
	}
    }
    free(str);
    str = NULL;

    /* keytab principal name */
    ret = krb5_ret_string(sp, &str);
    if (ret == 0 && str[0])
	ret = krb5_parse_name(context, str, &keytab_principal);
    if (ret) {
	*minor_status = ret;
	major_stat = GSS_S_FAILURE;
	goto out;
    }
    free(str);
    str = NULL;

    /* keytab principal */
    ret = krb5_ret_string(sp, &str);
    if (ret) {
	*minor_status = ret;
	major_stat =  GSS_S_FAILURE;
	goto out;
    }
    if (str[0]) {
	ret = krb5_kt_resolve(context, str, &keytab);
	if (ret) {
	    *minor_status = ret;
	    major_stat =  GSS_S_FAILURE;
	    goto out;
	}
    }
    free(str);
    str = NULL;

    major_stat = _gsskrb5_krb5_import_cred(minor_status, id, keytab_principal,
					   keytab, cred_handle);
out:
    if (id)
	krb5_cc_close(context, id);
    if (keytab_principal)
	krb5_free_principal(context, keytab_principal);
    if (keytab)
	krb5_kt_close(context, keytab);
    if (str)
	free(str);
    if (sp)
	krb5_storage_free(sp);

    return major_stat;
}


static OM_uint32
allowed_enctypes(OM_uint32 *minor_status,
		 krb5_context context,
		 gss_cred_id_t *cred_handle,
		 const gss_buffer_t value)
{
    OM_uint32 major_stat;
    krb5_error_code ret;
    size_t len, i, j;
    krb5_enctype *enctypes = NULL;
    krb5_storage *sp = NULL;
    gsskrb5_cred cred;

    if (cred_handle == NULL || *cred_handle == GSS_C_NO_CREDENTIAL) {
	*minor_status = 0;
	return GSS_S_FAILURE;
    }

    cred = (gsskrb5_cred)*cred_handle;

    if ((value->length % 4) != 0) {
	*minor_status = 0;
	major_stat = GSS_S_FAILURE;
	goto out;
    }

    len = value->length / 4;
    enctypes = malloc((len + 1) * 4);
    if (enctypes == NULL) {
	*minor_status = ENOMEM;
	major_stat = GSS_S_FAILURE;
	goto out;
    }

    sp = krb5_storage_from_mem(value->value, value->length);
    if (sp == NULL) {
	*minor_status = ENOMEM;
	major_stat = GSS_S_FAILURE;
	goto out;
    }

    for (j = 0, i = 0; i < len; i++) {
	uint32_t e;

	ret = krb5_ret_uint32(sp, &e);
	if (ret) {
	    *minor_status = ret;
	    major_stat =  GSS_S_FAILURE;
	    goto out;
	}
	if (krb5_enctype_valid(context, e) == 0)
	    enctypes[j++] = e;
    }
    enctypes[j++] = 0;

    if (cred->enctypes)
	free(cred->enctypes);
    cred->enctypes = enctypes;

    krb5_storage_free(sp);

    return GSS_S_COMPLETE;

out:
    if (sp)
	krb5_storage_free(sp);
    if (enctypes)
	free(enctypes);

    return major_stat;
}

static OM_uint32
no_ci_flags(OM_uint32 *minor_status,
	    krb5_context context,
	    gss_cred_id_t *cred_handle,
	    const gss_buffer_t value)
{
    gsskrb5_cred cred;

    if (cred_handle == NULL || *cred_handle == GSS_C_NO_CREDENTIAL) {
	*minor_status = 0;
	return GSS_S_FAILURE;
    }

    cred = (gsskrb5_cred)*cred_handle;
    cred->cred_flags |= GSS_CF_NO_CI_FLAGS;

    *minor_status = 0;
    return GSS_S_COMPLETE;

}

OM_uint32 GSSAPI_CALLCONV
_gsskrb5_set_cred_option
           (OM_uint32 *minor_status,
            gss_cred_id_t *cred_handle,
            const gss_OID desired_object,
            const gss_buffer_t value)
{
    krb5_context context;

    GSSAPI_KRB5_INIT (&context);

    if (value == GSS_C_NO_BUFFER) {
	*minor_status = EINVAL;
	return GSS_S_FAILURE;
    }

    if (gss_oid_equal(desired_object, GSS_KRB5_IMPORT_CRED_X))
	return import_cred(minor_status, context, cred_handle, value);

    if (gss_oid_equal(desired_object, GSS_KRB5_SET_ALLOWABLE_ENCTYPES_X))
	return allowed_enctypes(minor_status, context, cred_handle, value);

    if (gss_oid_equal(desired_object, GSS_KRB5_CRED_NO_CI_FLAGS_X))
	return no_ci_flags(minor_status, context, cred_handle, value);

    *minor_status = EINVAL;
    return GSS_S_FAILURE;
}
