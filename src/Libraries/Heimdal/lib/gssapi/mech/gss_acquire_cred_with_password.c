/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_acquire_cred_with_password(OM_uint32 * __nonnull minor_status,
			       __nonnull const gss_name_t desired_name,
			       __nonnull const gss_buffer_t password,
			       OM_uint32 time_req,
			       __nullable const gss_OID_set desired_mechs,
			       gss_cred_usage_t cred_usage,
			       __nullable gss_cred_id_t * __nonnull output_cred_handle,
			       __nullable gss_OID_set *__nullable actual_mechs,
			       OM_uint32 * __nullable time_rec)
{
    OM_uint32 major_status, tmp_minor;

    if (desired_mechs == GSS_C_NO_OID_SET) {
	major_status = gss_acquire_cred_ext(minor_status,
					    desired_name,
					    GSS_C_CRED_PASSWORD,
					    password,
					    time_req,
					    GSS_C_NO_OID,
					    cred_usage,
					    output_cred_handle);
	if (GSS_ERROR(major_status))
	    return major_status;
    } else {
	size_t i;
	struct _gss_cred *new_cred;

	new_cred = _gss_mg_alloc_cred();
	if (new_cred == NULL) {
	    *minor_status = ENOMEM;
	    return GSS_S_FAILURE;
	}
	HEIM_SLIST_INIT(&new_cred->gc_mc);

	for (i = 0; i < desired_mechs->count; i++) {
	    struct _gss_cred *tmp_cred = NULL;
	    struct _gss_mechanism_cred *mc;

	    major_status = gss_acquire_cred_ext(minor_status,
						desired_name,
						GSS_C_CRED_PASSWORD,
						password,
						time_req,
						&desired_mechs->elements[i],
						cred_usage,
						(gss_cred_id_t *)&tmp_cred);
	    if (GSS_ERROR(major_status))
		continue;

	    mc = HEIM_SLIST_FIRST(&tmp_cred->gc_mc);
	    if (mc) {
		HEIM_SLIST_REMOVE_HEAD(&tmp_cred->gc_mc, gmc_link);
		HEIM_SLIST_INSERT_HEAD(&new_cred->gc_mc, mc, gmc_link);
	    }

	    gss_release_cred(&tmp_minor, (gss_cred_id_t *)&tmp_cred);
	}

	if (!HEIM_SLIST_FIRST(&new_cred->gc_mc)) {
	    free(new_cred);
	    *minor_status = 0;
	    return GSS_S_NO_CRED;
	}

	*output_cred_handle = (gss_cred_id_t)new_cred;
    }

    if (actual_mechs != NULL || time_rec != NULL) {
	major_status = gss_inquire_cred(minor_status,
					*output_cred_handle,
					NULL,
					time_rec,
					NULL,
					actual_mechs);
	if (GSS_ERROR(major_status)) {
	    gss_release_cred(&tmp_minor, output_cred_handle);
	    return major_status;
	}
    }

    *minor_status = 0;
    return GSS_S_COMPLETE;
}
