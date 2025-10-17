/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
gss_compare_name(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_name_t name1_arg,
    __nonnull const gss_name_t name2_arg,
    int *__nonnull name_equal)
{
	struct _gss_name *name1 = (struct _gss_name *) name1_arg;
	struct _gss_name *name2 = (struct _gss_name *) name2_arg;

	/*
	 * First check the implementation-independant name if both
	 * names have one. Otherwise, try to find common mechanism
	 * names and compare them.
	 */
	if (name1->gn_value.value && name2->gn_value.value) {
		*name_equal = 1;
		if (!gss_oid_equal(&name1->gn_type, &name2->gn_type)) {
			*name_equal = 0;
		} else if (name1->gn_value.length != name2->gn_value.length ||
		    memcmp(name1->gn_value.value, name1->gn_value.value,
			name1->gn_value.length)) {
			*name_equal = 0;
		}
	} else {
		struct _gss_mechanism_name *mn1;
		struct _gss_mechanism_name *mn2;

		HEIM_SLIST_FOREACH(mn1, &name1->gn_mn, gmn_link) {
			OM_uint32 major_status;

			major_status = _gss_find_mn(minor_status, name2,
						    mn1->gmn_mech_oid, &mn2);
			if (major_status == GSS_S_COMPLETE && mn2) {
				return (mn1->gmn_mech->gm_compare_name(
						minor_status,
						mn1->gmn_name,
						mn2->gmn_name,
						name_equal));
			}
		}
		*name_equal = 0;
	}

	*minor_status = 0;
	return (GSS_S_COMPLETE);
}
