/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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

OM_uint32
_gss_acquire_mech_cred(OM_uint32 *__nonnull minor_status,
		       struct gssapi_mech_interface_desc *__nonnull m,
		       const struct _gss_mechanism_name *__nullable mn,
		       __nullable gss_const_OID credential_type,
		       const void *__nullable credential_data,
		       OM_uint32 time_req,
		       gss_const_OID __nullable desired_mech,
		       gss_cred_usage_t cred_usage,
		       struct _gss_mechanism_cred * __nullable * __nonnull output_cred_handle)
{
    OM_uint32 major_status;
    struct _gss_mechanism_cred *mc;
    gss_OID_set_desc set2;

    *output_cred_handle = NULL;

    mc = calloc(1, sizeof(struct _gss_mechanism_cred));
    if (mc == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    mc->gmc_mech = m;
    mc->gmc_mech_oid = &m->gm_mech_oid;

    set2.count = 1;
    set2.elements = mc->gmc_mech_oid;

    if (m->gm_acquire_cred_ext) {
	major_status = m->gm_acquire_cred_ext(minor_status,
					      mn ? mn->gmn_name : NULL,
					      credential_type,
					      credential_data,
					      time_req,
					      mc->gmc_mech_oid,
					      cred_usage,
					      &mc->gmc_cred);
	if (major_status)
	    _gss_mg_error(m, *minor_status);

    } else if (gss_oid_equal(credential_type, GSS_C_CRED_PASSWORD) &&
		m->gm_compat &&
		m->gm_compat->gmc_acquire_cred_with_password) {
	/*
	 * Shim for mechanisms that adhere to API-as-SPI and do not
	 * implement gss_acquire_cred_ext().
	 */

	major_status = m->gm_compat->gmc_acquire_cred_with_password(minor_status,
				mn ? mn->gmn_name : NULL,
				(const gss_buffer_t)credential_data,
				time_req,
				&set2,
				cred_usage,
				&mc->gmc_cred,
				NULL,
				NULL);
	if (major_status)
	    _gss_mg_error(m, *minor_status);

    } else if (credential_type == GSS_C_NO_OID) {
	major_status = m->gm_acquire_cred(minor_status,
					  mn ? mn->gmn_name : NULL,
					  time_req,
					  &set2,
					  cred_usage,
					  &mc->gmc_cred,
					  NULL,
					  NULL);
	if (major_status)
	    _gss_mg_error(m, *minor_status);

    } else {
	major_status = GSS_S_UNAVAILABLE;
	free(mc);
	mc= NULL;
    }

    *output_cred_handle = mc;
    return major_status;
}

OM_uint32
gss_acquire_cred_ext(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_name_t desired_name,
    __nonnull gss_const_OID credential_type,
    const void *__nonnull credential_data,
    OM_uint32 time_req,
    __nullable gss_const_OID desired_mech,
    gss_cred_usage_t cred_usage,
    __nonnull gss_cred_id_t *__nullable output_cred_handle)
{
    OM_uint32 major_status;
    struct _gss_name *name = (struct _gss_name *) desired_name;
    gssapi_mech_interface m;
    struct _gss_cred *cred;
    gss_OID_set_desc set, *mechs;
    size_t i;

    *minor_status = 0;
    if (output_cred_handle == NULL)
	return GSS_S_CALL_INACCESSIBLE_READ;

    _gss_load_mech();

    if (desired_mech != GSS_C_NO_OID) {
	int match = 0;

	gss_test_oid_set_member(minor_status, (gss_OID)desired_mech,
				_gss_mech_oids, &match);
	if (!match)
	    return GSS_S_BAD_MECH;

	set.count = 1;
	set.elements = (gss_OID)desired_mech;
	mechs = &set;
    } else
	mechs = _gss_mech_oids;

    cred = _gss_mg_alloc_cred();
    if (cred == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    /*
     * Return if no credential is created
     */

    major_status = GSS_S_NO_CRED;
    *minor_status = 0;

    for (i = 0; i < mechs->count; i++) {
	struct _gss_mechanism_name *mn = NULL;
	struct _gss_mechanism_cred *mc = NULL;
	gss_name_t desired_mech_name = GSS_C_NO_NAME;
	OM_uint32 major2, junk;

	m = __gss_get_mechanism(&mechs->elements[i]);
	if (!m)
	    continue;

	if (desired_name != GSS_C_NO_NAME) {
	    major2 = _gss_find_mn(&junk, name,
				  &mechs->elements[i], &mn);
	    if (major2 != GSS_S_COMPLETE)
		continue;

	    desired_mech_name = mn->gmn_name;
	}

	major_status = _gss_acquire_mech_cred(minor_status, m, mn,
					      credential_type, credential_data,
					      time_req, desired_mech, cred_usage,
					      &mc);
	if (GSS_ERROR(major_status))
	    continue;

	HEIM_SLIST_INSERT_HEAD(&cred->gc_mc, mc, gmc_link);
    }

    /*
     * If we didn't manage to create a single credential, return
     * the last mech's error.
     */
    if (!HEIM_SLIST_FIRST(&cred->gc_mc)) {
	free(cred);
	return major_status;
    }

    *output_cred_handle = (gss_cred_id_t) cred;
    *minor_status = 0;
    return GSS_S_COMPLETE;
}
