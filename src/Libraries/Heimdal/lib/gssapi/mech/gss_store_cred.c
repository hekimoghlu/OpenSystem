/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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
gss_store_cred(OM_uint32         *__nonnull minor_status,
	       __nonnull gss_cred_id_t      input_cred_handle,
	       gss_cred_usage_t             cred_usage,
	       __nullable const gss_OID     desired_mech,
	       OM_uint32         	    overwrite_cred,
	       OM_uint32         	    default_cred,
	       __nullable gss_OID_set       *__nullable elements_stored,
	       gss_cred_usage_t  *__nullable cred_usage_stored)
{
    struct _gss_cred *cred = (struct _gss_cred *) input_cred_handle;
    struct _gss_mechanism_cred *mc;
    OM_uint32 maj, junk;

    if (minor_status == NULL)
	return GSS_S_FAILURE;
    if (elements_stored)
	*elements_stored = NULL;
    if (cred_usage_stored)
	*cred_usage_stored = 0;

    if (cred == NULL)
	return GSS_S_NO_CONTEXT;

    if (elements_stored) {
	maj = gss_create_empty_oid_set(minor_status, elements_stored);
	if (maj != GSS_S_COMPLETE)
	    return maj;
    }

    HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {
	gssapi_mech_interface m = mc->gmc_mech;

	if (m == NULL || m->gm_store_cred == NULL)
	    continue;

	if (desired_mech) {
	    maj = gss_oid_equal(&m->gm_mech_oid, desired_mech);
	    if (maj != 0)
		continue;
	}

	maj = (m->gm_store_cred)(minor_status, mc->gmc_cred,
				 cred_usage, desired_mech, overwrite_cred,
				 default_cred, NULL, cred_usage_stored);
	if (maj != GSS_S_COMPLETE) {
	    gss_release_oid_set(&junk, elements_stored);
	    return maj;
	}

	if (elements_stored) {
	    gss_add_oid_set_member(&junk,
				   &m->gm_mech_oid,
				   elements_stored);
	}

    }
    return GSS_S_COMPLETE;
}
