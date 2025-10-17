/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 23, 2023.
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
gss_set_cred_option (OM_uint32 *__nonnull minor_status,
		     __nullable gss_cred_id_t * __nullable cred_handle,
		     __nonnull const gss_OID object,
		     __nullable const gss_buffer_t value)
{
	struct _gss_cred *cred = (struct _gss_cred *) *cred_handle;
	OM_uint32	major_status = GSS_S_COMPLETE;
	struct _gss_mechanism_cred *mc;
	int one_ok = 0;

	*minor_status = 0;

	_gss_load_mech();

	if (cred == NULL) {
		struct _gss_mech_switch *m;

		cred = _gss_mg_alloc_cred();
		if (cred == NULL)
		    return GSS_S_FAILURE;

		HEIM_SLIST_FOREACH(m, &_gss_mechs, gm_link) {

			if (m->gm_mech.gm_set_cred_option == NULL)
				continue;

			mc = malloc(sizeof(*mc));
			if (mc == NULL) {
			    *cred_handle = (gss_cred_id_t)cred;
			    gss_release_cred(minor_status, cred_handle);
			    *minor_status = ENOMEM;
			    return GSS_S_FAILURE;
			}

			mc->gmc_mech = &m->gm_mech;
			mc->gmc_mech_oid = &m->gm_mech_oid;
			mc->gmc_cred = GSS_C_NO_CREDENTIAL;

			major_status = m->gm_mech.gm_set_cred_option(
			    minor_status, &mc->gmc_cred, object, value);

			if (major_status) {
				free(mc);
				continue;
			}
			one_ok = 1;
			HEIM_SLIST_INSERT_HEAD(&cred->gc_mc, mc, gmc_link);
		}
		*cred_handle = (gss_cred_id_t)cred;
		if (!one_ok) {
			OM_uint32 junk;
			gss_release_cred(&junk, cred_handle);
		}
	} else {
		gssapi_mech_interface	m;

		HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {
			m = mc->gmc_mech;

			if (m == NULL)
				return GSS_S_BAD_MECH;

			if (m->gm_set_cred_option == NULL)
				continue;

			major_status = m->gm_set_cred_option(minor_status,
			    &mc->gmc_cred, object, value);
			if (major_status == GSS_S_COMPLETE)
				one_ok = 1;
			else
				_gss_mg_error(m, *minor_status);

		}
	}
	if (one_ok) {
		*minor_status = 0;
		return GSS_S_COMPLETE;
	}
	return major_status;
}

