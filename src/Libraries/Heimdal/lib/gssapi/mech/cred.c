/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 5, 2025.
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
#include "heim_threads.h"
#include "heimbase.h"


void
_gss_mg_release_cred(struct _gss_cred *__nonnull cred)
{
	struct _gss_mechanism_cred *mc;
	OM_uint32 junk;

	while (HEIM_SLIST_FIRST(&cred->gc_mc)) {
		mc = HEIM_SLIST_FIRST(&cred->gc_mc);
		HEIM_SLIST_REMOVE_HEAD(&cred->gc_mc, gmc_link);
		mc->gmc_mech->gm_release_cred(&junk, &mc->gmc_cred);
		free(mc);
	}
	free(cred);
}

struct _gss_cred * __nullable
_gss_mg_alloc_cred(void)
{
	struct _gss_cred *cred;
	cred = malloc(sizeof(struct _gss_cred));
	if (!cred)
		return NULL;
	HEIM_SLIST_INIT(&cred->gc_mc);

	return cred;
}

void
_gss_mg_check_credential(gss_cred_id_t __nullable credential)
{
	if (credential == NULL) return;
}

__nullable gss_name_t
_gss_cred_copy_name( OM_uint32 *__nonnull minor_status,
		    __nonnull gss_cred_id_t credential,
		    __nullable gss_const_OID mech)
{
	struct _gss_cred *cred = (struct _gss_cred *)credential;
	struct _gss_mechanism_cred *mc;
	struct _gss_name *name;
	OM_uint32 major_status;

	name = _gss_create_name(NULL, NULL);
	if (name == NULL)
		return NULL;

	HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {
		struct _gss_mechanism_name *mn;
		gss_name_t mc_name;
		
		if (mech && !gss_oid_equal(mech, mc->gmc_mech_oid))
			continue;

		major_status = mc->gmc_mech->gm_inquire_cred(minor_status,
			mc->gmc_cred, &mc_name, NULL, NULL, NULL);
		if (major_status)
			continue;

		mn = malloc(sizeof(struct _gss_mechanism_name));
		if (!mn) {
			mc->gmc_mech->gm_release_name(minor_status, &mc_name);
			continue;
		}
		mn->gmn_mech = mc->gmc_mech;
		mn->gmn_mech_oid = mc->gmc_mech_oid;
		mn->gmn_name = mc_name;
		HEIM_SLIST_INSERT_HEAD(&name->gn_mn, mn, gmn_link);
	}
	if (HEIM_SLIST_EMPTY(&name->gn_mn)) {
		_gss_mg_release_name(name);
		return NULL;
	}

	return (gss_name_t)name;
}
