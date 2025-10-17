/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
gss_inquire_mechs_for_name(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_name_t input_name,
    __nullable gss_OID_set * __nonnull mech_types)
{
	OM_uint32		major_status;
	struct _gss_name	*name = (struct _gss_name *) input_name;
	struct _gss_mech_switch	*m;
	gss_OID_set		name_types;
	int			present;

	*minor_status = 0;

	_gss_load_mech();

	major_status = gss_create_empty_oid_set(minor_status, mech_types);
	if (major_status)
		return (major_status);

	/*
	 * We go through all the loaded mechanisms and see if this
	 * name's type is supported by the mechanism. If it is, add
	 * the mechanism to the set.
	 */
	HEIM_SLIST_FOREACH(m, &_gss_mechs, gm_link) {
		major_status = gss_inquire_names_for_mech(minor_status,
		    &m->gm_mech_oid, &name_types);
		if (major_status) {
			gss_release_oid_set(minor_status, mech_types);
			return (major_status);
		}
		gss_test_oid_set_member(minor_status,
		    &name->gn_type, name_types, &present);
		gss_release_oid_set(minor_status, &name_types);
		if (present) {
			major_status = gss_add_oid_set_member(minor_status,
			    &m->gm_mech_oid, mech_types);
			if (major_status) {
				gss_release_oid_set(minor_status, mech_types);
				return (major_status);
			}
		}
	}

	return (GSS_S_COMPLETE);
}
