/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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

/**
 * Add a oid to the oid set, function does not make a copy of the oid,
 * so the pointer to member_oid needs to be stable for the whole time
 * oid_set is used.
 *
 * If there is a duplicate member of the oid, the new member is not
 * added to to the set.
 *
 * @param minor_status minor status code.
 * @param member_oid member to add to the oid set
 * @param oid_set oid set to add the member too
 *
 * @returns a gss_error code, see gss_display_status() about printing
 *          the error code.
 *
 * @ingroup gssapi
 */

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_add_oid_set_member (OM_uint32 * __nonnull minor_status,
			__nonnull gss_const_OID member_oid,
			__nonnull gss_OID_set * __nonnull oid_set)
{
    gss_OID tmp;
    size_t n;
    OM_uint32 res;
    int present;

    res = gss_test_oid_set_member(minor_status, member_oid, *oid_set, &present);
    if (res != GSS_S_COMPLETE)
	return res;

    if (present) {
	*minor_status = 0;
	return GSS_S_COMPLETE;
    }

    n = (*oid_set)->count + 1;
    tmp = realloc ((*oid_set)->elements, n * sizeof(gss_OID_desc));
    if (tmp == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }
    (*oid_set)->elements = tmp;
    (*oid_set)->count = n;
    (*oid_set)->elements[n-1] = *member_oid;
    *minor_status = 0;
    return GSS_S_COMPLETE;
}
