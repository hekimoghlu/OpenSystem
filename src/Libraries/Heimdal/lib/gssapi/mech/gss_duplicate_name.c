/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
gss_duplicate_name(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_name_t src_name,
    __nullable gss_name_t * __nonnull dest_name)
{
	OM_uint32		major_status;
	struct _gss_name	*name = (struct _gss_name *) src_name;
	struct _gss_name	*new_name;
	struct _gss_mechanism_name *mn;

	_gss_mg_check_name(src_name);

	*minor_status = 0;
	*dest_name = GSS_C_NO_NAME;

	/*
	 * If this name has a value (i.e. it didn't come from
	 * gss_canonicalize_name(), we re-import the thing. Otherwise,
	 * we make copy of each mech names.
	 */
	if (name->gn_value.value) {
		major_status = gss_import_name(minor_status,
		    &name->gn_value, &name->gn_type, dest_name);
		if (major_status != GSS_S_COMPLETE)
			return (major_status);
		new_name = (struct _gss_name *) *dest_name;

		HEIM_SLIST_FOREACH(mn, &name->gn_mn, gmn_link) {
		    struct _gss_mechanism_name *mn2;
		    _gss_find_mn(minor_status, new_name,
				 mn->gmn_mech_oid, &mn2);
		}
	} else {
		new_name = _gss_create_name(NULL, NULL);
		if (!new_name) {
			*minor_status = ENOMEM;
			return (GSS_S_FAILURE);
		}
		*dest_name = (gss_name_t) new_name;

		HEIM_SLIST_FOREACH(mn, &name->gn_mn, gmn_link) {
			struct _gss_mechanism_name *new_mn;

			new_mn = malloc(sizeof(*new_mn));
			if (!new_mn) {
				*minor_status = ENOMEM;
				return GSS_S_FAILURE;
			}
			new_mn->gmn_mech = mn->gmn_mech;
			new_mn->gmn_mech_oid = mn->gmn_mech_oid;

			major_status =
			    mn->gmn_mech->gm_duplicate_name(minor_status,
				mn->gmn_name, &new_mn->gmn_name);
			if (major_status != GSS_S_COMPLETE) {
				free(new_mn);
				continue;
			}
			HEIM_SLIST_INSERT_HEAD(&new_name->gn_mn, new_mn, gmn_link);
		}

	}

	return (GSS_S_COMPLETE);
}
