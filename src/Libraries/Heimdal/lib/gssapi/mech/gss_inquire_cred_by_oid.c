/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
gss_inquire_cred_by_oid (OM_uint32 *__nonnull minor_status,
			 __nonnull const gss_cred_id_t cred_handle,
			 __nonnull const gss_OID desired_object,
			 __nullable gss_buffer_set_t * __nonnull data_set)
{
	struct _gss_cred *cred = (struct _gss_cred *) cred_handle;
	OM_uint32		status = GSS_S_COMPLETE;
	struct _gss_mechanism_cred *mc;
	gssapi_mech_interface	m;
	gss_buffer_set_t set = GSS_C_NO_BUFFER_SET;

	*minor_status = 0;
	*data_set = GSS_C_NO_BUFFER_SET;

	if (cred == NULL)
		return GSS_S_NO_CRED;

	status = GSS_S_FAILURE;

	HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {
		gss_buffer_set_t rset = GSS_C_NO_BUFFER_SET;
		size_t i;

		m = mc->gmc_mech;
		if (m == NULL) {
	       		gss_release_buffer_set(minor_status, &set);
			*minor_status = 0;
			return GSS_S_BAD_MECH;
		}

		if (m->gm_inquire_cred_by_oid == NULL)
			continue;

		status = m->gm_inquire_cred_by_oid(minor_status,
		    mc->gmc_cred, desired_object, &rset);
		if (status != GSS_S_COMPLETE) {
			_gss_mg_error(m, *minor_status);
			continue;
		}

		for (i = 0; rset != NULL && i < rset->count; i++) {
			status = gss_add_buffer_set_member(minor_status,
			     &rset->elements[i], &set);
			if (status != GSS_S_COMPLETE)
				break;
		}
		gss_release_buffer_set(minor_status, &rset);
	}

	if (set == GSS_C_NO_BUFFER_SET) {
		if (status == GSS_S_COMPLETE)
			status = GSS_S_FAILURE;
		return status;
	}

	*data_set = set;
	*minor_status = 0;

	return GSS_S_COMPLETE;
}

