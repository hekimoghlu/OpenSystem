/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
gss_inquire_cred_by_mech(OM_uint32 * __nonnull minor_status,
    __nullable const gss_cred_id_t cred_handle,
    __nonnull const gss_OID mech_type,
    __nullable gss_name_t * __nullable cred_name,
    OM_uint32 *__nullable initiator_lifetime,
    OM_uint32 *__nullable acceptor_lifetime,
    gss_cred_usage_t *__nullable cred_usage)
{
	OM_uint32 major_status;
	gssapi_mech_interface m;
	struct _gss_mechanism_cred *mcp;
	gss_cred_id_t mc;
	gss_name_t mn;
	struct _gss_name *name;

	*minor_status = 0;
	if (cred_name)
	    *cred_name = GSS_C_NO_NAME;
	if (initiator_lifetime)
	    *initiator_lifetime = 0;
	if (acceptor_lifetime)
	    *acceptor_lifetime = 0;
	if (cred_usage)
	    *cred_usage = 0;

	m = __gss_get_mechanism(mech_type);
	if (m == NULL || m->gm_inquire_cred_by_mech == NULL)
		return (GSS_S_NO_CRED);

	if (cred_handle != GSS_C_NO_CREDENTIAL) {
		struct _gss_cred *cred = (struct _gss_cred *) cred_handle;
		HEIM_SLIST_FOREACH(mcp, &cred->gc_mc, gmc_link)
			if (mcp->gmc_mech == m)
				break;
		if (!mcp)
			return (GSS_S_NO_CRED);
		mc = mcp->gmc_cred;
	} else {
		mc = GSS_C_NO_CREDENTIAL;
	}

	major_status = m->gm_inquire_cred_by_mech(minor_status, mc, mech_type,
	    &mn, initiator_lifetime, acceptor_lifetime, cred_usage);
	if (major_status != GSS_S_COMPLETE) {
		_gss_mg_error(m, *minor_status);
		return (major_status);
	}

	if (cred_name) {
	    name = _gss_create_name(mn, m);
	    if (!name) {
		m->gm_release_name(minor_status, &mn);
		return (GSS_S_NO_CRED);
	    }
	    *cred_name = (gss_name_t) name;
	} else
	    m->gm_release_name(minor_status, &mn);


	return (GSS_S_COMPLETE);
}
