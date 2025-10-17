/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
#include "mech_locl.h"
#include <krb5.h>

/*
 * format: any number of:
 *     mech-len: int32
 *     mech-data: char * (not alligned)
 *     cred-len: int32
 *     cred-data char * (not alligned)
*/

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_export_cred(OM_uint32 * __nonnull minor_status,
		__nonnull gss_cred_id_t cred_handle,
		__nonnull gss_buffer_t token)
{
    struct _gss_cred *cred = (struct _gss_cred *)cred_handle;
    struct _gss_mechanism_cred *mc;
    gss_buffer_desc buffer;
    krb5_error_code ret;
    krb5_storage *sp;
    OM_uint32 major;
    krb5_data data;

    _mg_buffer_zero(token);

    if (cred == NULL) {
	*minor_status = 0;
	return GSS_S_NO_CRED;
    }

    HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {
	if (mc->gmc_mech->gm_export_cred == NULL) {
	    *minor_status = 0;
	    gss_mg_set_error_string(&mc->gmc_mech->gm_mech_oid,
				    GSS_S_NO_CRED, *minor_status,
				    "Credential doesn't support exporting");
	    return GSS_S_NO_CRED;
	}
    }

    sp = krb5_storage_emem();
    if (sp == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {
	krb5_ssize_t sret;
	
	major = mc->gmc_mech->gm_export_cred(minor_status,
					     mc->gmc_cred, &buffer);
	if (major) {
	    krb5_storage_free(sp);
	    return major;
	}

	if (buffer.length) {
	    sret = krb5_storage_write(sp, buffer.value, buffer.length);
	    if (sret < 0 || (size_t)sret != buffer.length) {
		gss_release_buffer(minor_status, &buffer);
		krb5_storage_free(sp);
		*minor_status = EINVAL;
		return GSS_S_FAILURE;
	    }
	}
	gss_release_buffer(minor_status, &buffer);
    }

    ret = krb5_storage_to_data(sp, &data);
    krb5_storage_free(sp);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }

    if (data.length == 0)  {
	*minor_status = 0;
	gss_mg_set_error_string(GSS_C_NO_OID,
				GSS_S_NO_CRED, *minor_status,
				"Credential was not exportable");
	return GSS_S_NO_CRED;
    }

    token->value = data.data;
    token->length = data.length;

    return GSS_S_COMPLETE;
}

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_import_cred(OM_uint32 * __nonnull minor_status,
		__nonnull gss_buffer_t token,
		__nullable gss_cred_id_t * __nonnull cred_handle)
{
    gssapi_mech_interface m;
    krb5_error_code ret;
    struct _gss_cred *cred;
    krb5_storage *sp = NULL;
    OM_uint32 major, junk;
    krb5_data data;

    *cred_handle = GSS_C_NO_CREDENTIAL;

    if (token->length == 0) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    sp = krb5_storage_from_readonly_mem(token->value, token->length);
    if (sp == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    cred = _gss_mg_alloc_cred();
    if (cred == NULL) {
	krb5_storage_free(sp);
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    *cred_handle = (gss_cred_id_t)cred;

    while(1) {
	struct _gss_mechanism_cred *mc;
	gss_buffer_desc buffer;
	gss_cred_id_t mcred;
	gss_OID_desc oid;

	ret = krb5_ret_data(sp, &data);
	if (ret == HEIM_ERR_EOF) {
	    break;
	} else if (ret) {
	    *minor_status = ret;
	    major = GSS_S_FAILURE;
	    goto out;
	}
	oid.elements = data.data;
	oid.length = (OM_uint32)data.length;

	m = __gss_get_mechanism(&oid);
	krb5_data_free(&data);
	if (!m) {
	    *minor_status = 0;
	    major = GSS_S_BAD_MECH;
	    goto out;
	}

	if (m->gm_import_cred == NULL) {
	    *minor_status = 0;
	    major = GSS_S_BAD_MECH;
	    goto out;
	}

	ret = krb5_ret_data(sp, &data);
	if (ret) {
	    *minor_status = ret;
	    major = GSS_S_FAILURE;
	    goto out;
	}

	buffer.value = data.data;
	buffer.length = data.length;

	major = m->gm_import_cred(minor_status,
				  &buffer, &mcred);
	krb5_data_free(&data);
	if (major) {
	    goto out;
	}

	mc = malloc(sizeof(struct _gss_mechanism_cred));
	if (mc == NULL) {
	    *minor_status = EINVAL;
	    major = GSS_S_FAILURE;
	    goto out;
	}

	mc->gmc_mech = m;
	mc->gmc_mech_oid = &m->gm_mech_oid;
	mc->gmc_cred = mcred;

	HEIM_SLIST_INSERT_HEAD(&cred->gc_mc, mc, gmc_link);
    }
    krb5_storage_free(sp);
    sp = NULL;

    if (HEIM_SLIST_EMPTY(&cred->gc_mc)) {
	major = GSS_S_NO_CRED;
	goto out;
    }

    return GSS_S_COMPLETE;

 out:
    if (sp)
	krb5_storage_free(sp);

    gss_release_cred(&junk, cred_handle);

    return major;

}

OM_uint32
gss_cred_label_get(OM_uint32 * __nonnull min_stat,
		   __nonnull gss_cred_id_t cred_handle,
		   const char * __nonnull label,
		   __nonnull gss_buffer_t value)
{
    struct _gss_cred *cred = (struct _gss_cred *)cred_handle;
    struct _gss_mechanism_cred *mc;
    OM_uint32 maj_stat;

    *min_stat = 0;
    _mg_buffer_zero(value);

    HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {

	if (mc->gmc_mech->gm_cred_label_get == NULL)
	    continue;

	maj_stat = mc->gmc_mech->gm_cred_label_get(min_stat, mc->gmc_cred,
						   label, value);
	if (maj_stat == GSS_S_COMPLETE)
	    return GSS_S_COMPLETE;
    }

    return GSS_S_UNAVAILABLE;
}

OM_uint32
gss_cred_label_set(OM_uint32 * __nonnull min_stat,
		   __nonnull gss_cred_id_t cred_handle,
		   const char * __nonnull label,
		   __nullable gss_buffer_t value)
{
    struct _gss_cred *cred = (struct _gss_cred *)cred_handle;
    struct _gss_mechanism_cred *mc;

    *min_stat = 0;

    HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {

	if (mc->gmc_mech->gm_cred_label_set == NULL)
	    continue;

	(void)mc->gmc_mech->gm_cred_label_set(min_stat, mc->gmc_cred,
					      label, value);
    }

    return GSS_S_COMPLETE;
}
