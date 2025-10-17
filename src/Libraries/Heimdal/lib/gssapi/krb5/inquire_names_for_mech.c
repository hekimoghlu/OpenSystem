/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#include "gsskrb5_locl.h"

static gss_OID name_list[] = {
    GSS_C_NT_HOSTBASED_SERVICE,
    GSS_C_NT_USER_NAME,
    GSS_KRB5_NT_PRINCIPAL_NAME,
    GSS_C_NT_EXPORT_NAME,
    NULL
};

OM_uint32 GSSAPI_CALLCONV _gsskrb5_inquire_names_for_mech (
            OM_uint32 * minor_status,
            gss_const_OID mechanism,
            gss_OID_set * name_types
           )
{
    OM_uint32 ret, junk;
    int i;

    *minor_status = 0;

    if (gss_oid_equal(mechanism, GSS_KRB5_MECHANISM) == 0 &&
	gss_oid_equal(mechanism, GSS_C_NULL_OID) == 0) {
	*name_types = GSS_C_NO_OID_SET;
	return GSS_S_BAD_MECH;
    }

    ret = gss_create_empty_oid_set(minor_status, name_types);
    if (ret != GSS_S_COMPLETE)
	return ret;

    for (i = 0; name_list[i] != NULL; i++) {
	ret = gss_add_oid_set_member(minor_status,
				     name_list[i],
				     name_types);
	if (ret != GSS_S_COMPLETE)
	    break;
    }

    if (ret != GSS_S_COMPLETE)
	gss_release_oid_set(&junk, name_types);

    return GSS_S_COMPLETE;
}
