/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
gss_inquire_names_for_mech(OM_uint32 *__nonnull minor_status,
    __nonnull gss_const_OID mechanism,
    __nullable gss_OID_set * __nonnull name_types)
{
	OM_uint32 major_status;
	gssapi_mech_interface m = __gss_get_mechanism(mechanism);

	*minor_status = 0;
	*name_types = GSS_C_NO_OID_SET;
	if (!m)
		return (GSS_S_BAD_MECH);

	/*
	 * If the implementation can do it, ask it for a list of
	 * names, otherwise fake it.
	 */
	if (m->gm_inquire_names_for_mech) {
		return (m->gm_inquire_names_for_mech(minor_status,
			    mechanism, name_types));
	} else {
		major_status = gss_create_empty_oid_set(minor_status,
		    name_types);
		if (major_status)
			return (major_status);
		major_status = gss_add_oid_set_member(minor_status,
		    GSS_C_NT_HOSTBASED_SERVICE, name_types);
		if (major_status) {
			OM_uint32 junk;
			gss_release_oid_set(&junk, name_types);
			return (major_status);
		}
		major_status = gss_add_oid_set_member(minor_status,
		    GSS_C_NT_USER_NAME, name_types);
		if (major_status) {
			OM_uint32 junk;
			gss_release_oid_set(&junk, name_types);
			return (major_status);
		}
	}

	return (GSS_S_COMPLETE);
}
