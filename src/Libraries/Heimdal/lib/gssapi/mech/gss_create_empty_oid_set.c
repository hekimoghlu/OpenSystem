/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
gss_create_empty_oid_set(OM_uint32 *__nonnull minor_status,
    __nullable gss_OID_set *__nonnull oid_set)
{
	gss_OID_set set;

	*minor_status = 0;
	*oid_set = GSS_C_NO_OID_SET;

	set = malloc(sizeof(gss_OID_set_desc));
	if (!set) {
		*minor_status = ENOMEM;
		return (GSS_S_FAILURE);
	}

	set->count = 0;
	set->elements = 0;
	*oid_set = set;

	return (GSS_S_COMPLETE);
}
