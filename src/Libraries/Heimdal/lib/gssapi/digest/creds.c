/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
#include "gssdigest.h"
#include "heimcred.h"

OM_uint32 _gss_scram_inquire_cred
           (OM_uint32 * minor_status,
            const gss_cred_id_t cred_handle,
            gss_name_t * name,
            OM_uint32 * lifetime,
            gss_cred_usage_t * cred_usage,
            gss_OID_set * mechanisms
           )
{
    OM_uint32 ret, junk;

    *minor_status = 0;

    if (cred_handle == NULL)
	return GSS_S_NO_CRED;

    if (name) {
	ret = _gss_scram_duplicate_name(minor_status,
				       (gss_name_t)cred_handle,
				       name);
	if (ret)
	    goto out;
    }
    if (lifetime)
	*lifetime = GSS_C_INDEFINITE;
    if (cred_usage)
	*cred_usage = 0;
    if (mechanisms)
	*mechanisms = GSS_C_NO_OID_SET;

    if (cred_handle == GSS_C_NO_CREDENTIAL)
	return GSS_S_NO_CRED;

    if (mechanisms) {
        ret = gss_create_empty_oid_set(minor_status, mechanisms);
        if (ret)
	    goto out;
	ret = gss_add_oid_set_member(minor_status,
				     GSS_SCRAM_MECHANISM,
				     mechanisms);
        if (ret)
	    goto out;
    }

    return GSS_S_COMPLETE;
out:
    gss_release_oid_set(&junk, mechanisms);
    return ret;
}

OM_uint32
_gss_scram_destroy_cred(OM_uint32 *minor_status,
		       gss_cred_id_t *cred_handle)
{
    if (cred_handle == NULL || *cred_handle == GSS_C_NO_CREDENTIAL)
	return GSS_S_COMPLETE;

#ifdef HAVE_KCM
    krb5_error_code ret;
    krb5_storage *request, *response;
    krb5_data response_data;
    krb5_context context;

    ret = krb5_init_context(&context);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }

    ret = krb5_kcm_storage_request(context, KCM_OP_DEL_SCRAM_CRED, &request);
    if (ret)
	goto out;

    ret = krb5_store_stringz(request, (char *)*cred_handle);
    if (ret)
	goto out;

    ret = krb5_kcm_call(context, request, &response, &response_data);
    if (ret)
	goto out;

    krb5_storage_free(request);
    krb5_storage_free(response);
    krb5_data_free(&response_data);

 out:
    krb5_free_context(context);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }
#else
    CFUUIDRef uuid_cfuuid = NULL;
    scram_cred cred = (scram_cred)cred_handle;

    CFUUIDBytes cfuuid;
    memcpy(&cfuuid, &cred->uuid, sizeof(cred->uuid));
    uuid_cfuuid = CFUUIDCreateFromUUIDBytes(kCFAllocatorDefault, cfuuid );
    HeimCredDeleteByUUID(uuid_cfuuid);
    CFRelease(uuid_cfuuid);
#endif
    return _gss_scram_release_cred(minor_status, cred_handle);
}
