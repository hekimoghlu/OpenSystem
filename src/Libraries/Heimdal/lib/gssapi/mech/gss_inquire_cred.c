/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 7, 2022.
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

#define AUSAGE 1
#define IUSAGE 2

static void
updateusage(gss_cred_usage_t usage, int *usagemask)
{
    if (usage == GSS_C_BOTH)
	*usagemask |= AUSAGE | IUSAGE;
    else if (usage == GSS_C_ACCEPT)
	*usagemask |= AUSAGE;
    else if (usage == GSS_C_INITIATE)
	*usagemask |= IUSAGE;
}

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_inquire_cred(OM_uint32  * __nonnull  minor_status,
    __nullable const gss_cred_id_t cred_handle,
    __nullable gss_name_t * __nullable name_ret,
    OM_uint32 * __nullable lifetime,
    gss_cred_usage_t  * __nullable cred_usage,
    __nullable gss_OID_set * __nullable mechanisms)
{
	OM_uint32 major_status;
	struct _gss_mech_switch *m;
	struct _gss_cred *cred = (struct _gss_cred *) cred_handle;
	struct _gss_name *name;
	struct _gss_mechanism_name *mn;
	OM_uint32 min_lifetime;
	int found = 0;
	int usagemask = 0;
	gss_cred_usage_t usage;

	_gss_load_mech();

	*minor_status = 0;
	if (name_ret)
		*name_ret = GSS_C_NO_NAME;
	if (lifetime)
		*lifetime = 0;
	if (cred_usage)
		*cred_usage = 0;
	if (mechanisms)
		*mechanisms = GSS_C_NO_OID_SET;

	if (name_ret) {
		name = _gss_create_name(NULL, NULL);
		if (name == NULL) {
			*minor_status = ENOMEM;
			return (GSS_S_FAILURE);
		}
	} else {
		name = NULL;
	}

	if (mechanisms) {
		major_status = gss_create_empty_oid_set(minor_status,
		    mechanisms);
		if (major_status) {
			if (name) free(name);
			return (major_status);
		}
	}

	min_lifetime = GSS_C_INDEFINITE;
	if (cred) {
		struct _gss_mechanism_cred *mc;

		HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {
			gss_name_t mc_name;
			OM_uint32 mc_lifetime;

			major_status = mc->gmc_mech->gm_inquire_cred(minor_status,
			    mc->gmc_cred, &mc_name, &mc_lifetime, &usage, NULL);
			if (major_status)
				continue;

			updateusage(usage, &usagemask);
			if (name) {
				mn = malloc(sizeof(struct _gss_mechanism_name));
				if (!mn) {
					mc->gmc_mech->gm_release_name(minor_status,
					    &mc_name);
					continue;
				}
				mn->gmn_mech = mc->gmc_mech;
				mn->gmn_mech_oid = mc->gmc_mech_oid;
				mn->gmn_name = mc_name;
				HEIM_SLIST_INSERT_HEAD(&name->gn_mn, mn, gmn_link);
			} else {
				mc->gmc_mech->gm_release_name(minor_status,
				    &mc_name);
			}

			if (mc_lifetime < min_lifetime)
				min_lifetime = mc_lifetime;

			if (mechanisms)
				gss_add_oid_set_member(minor_status,
				    mc->gmc_mech_oid, mechanisms);
			found++;
		}
	} else {
		HEIM_SLIST_FOREACH(m, &_gss_mechs, gm_link) {
			gss_name_t mc_name;
			OM_uint32 mc_lifetime;

			if (m->gm_mech.gm_inquire_cred == NULL)
				continue;

			major_status = m->gm_mech.gm_inquire_cred(minor_status,
			    GSS_C_NO_CREDENTIAL, &mc_name, &mc_lifetime,
			    &usage, NULL);
			if (major_status)
				continue;

			updateusage(usage, &usagemask);
			if (name && mc_name) {
				mn = malloc(
					sizeof(struct _gss_mechanism_name));
				if (!mn) {
					m->gm_mech.gm_release_name(
						minor_status, &mc_name);
					continue;
				}
				mn->gmn_mech = &m->gm_mech;
				mn->gmn_mech_oid = &m->gm_mech_oid;
				mn->gmn_name = mc_name;
				HEIM_SLIST_INSERT_HEAD(&name->gn_mn, mn, gmn_link);
			} else if (mc_name) {
				m->gm_mech.gm_release_name(minor_status,
				    &mc_name);
			}

			if (mc_lifetime < min_lifetime)
				min_lifetime = mc_lifetime;

			if (mechanisms)
				gss_add_oid_set_member(minor_status,
				    &m->gm_mech_oid, mechanisms);
			found++;
		}
	}

	if (found == 0 || min_lifetime == 0) {
		gss_name_t n = (gss_name_t)name;
		if (n)
			gss_release_name(minor_status, &n);
		gss_release_oid_set(minor_status, mechanisms);
		*minor_status = 0;
		if (min_lifetime == 0)
			return (GSS_S_CREDENTIALS_EXPIRED);
		return (GSS_S_NO_CRED);
	}

	*minor_status = 0;
	if (name_ret)
		*name_ret = (gss_name_t) name;
	if (lifetime)
		*lifetime = min_lifetime;
	if (cred_usage) {
		if ((usagemask & (AUSAGE|IUSAGE)) == (AUSAGE|IUSAGE))
			*cred_usage = GSS_C_BOTH;
		else if (usagemask & IUSAGE)
			*cred_usage = GSS_C_INITIATE;
		else if (usagemask & AUSAGE)
			*cred_usage = GSS_C_ACCEPT;
	}

	return (GSS_S_COMPLETE);
}
