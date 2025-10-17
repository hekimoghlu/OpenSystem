/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
gss_indicate_mechs(OM_uint32 *__nonnull minor_status,
    __nullable gss_OID_set * __nonnull mech_set)
{
	struct _gss_mech_switch *m;
	OM_uint32 major_status, junk;
	gss_OID_set set;
	size_t i;

	_gss_load_mech();

	major_status = gss_create_empty_oid_set(minor_status, mech_set);
	if (major_status)
		return (major_status);

	HEIM_SLIST_FOREACH(m, &_gss_mechs, gm_link) {
		if (m->gm_mech.gm_indicate_mechs) {
			major_status = m->gm_mech.gm_indicate_mechs(
			    minor_status, &set);
			if (major_status)
				continue;
			major_status = GSS_S_COMPLETE;
			for (i = 0; i < set->count; i++) {
				major_status = gss_add_oid_set_member(
				    minor_status, &set->elements[i], mech_set);
				if (major_status)
					break;
			}
			gss_release_oid_set(minor_status, &set);
		} else {
			major_status = gss_add_oid_set_member(
			    minor_status, &m->gm_mech_oid, mech_set);
		}
		if (major_status)
			break;
	}
	if (major_status)
		gss_release_oid_set(&junk, mech_set);

	*minor_status = 0;
	return major_status;
}
