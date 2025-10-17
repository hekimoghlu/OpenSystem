/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
gss_duplicate_oid (
        OM_uint32 *__nonnull minor_status,
	__nonnull gss_OID src_oid,
	__nullable gss_OID * __nonnull dest_oid
     )
{
    *minor_status = 0;

    if (src_oid == GSS_C_NO_OID) {
	*dest_oid = GSS_C_NO_OID;
	return GSS_S_COMPLETE;
    }

    *dest_oid = malloc(sizeof(**dest_oid));
    if (*dest_oid == GSS_C_NO_OID) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    (*dest_oid)->elements = malloc(src_oid->length);
    if ((*dest_oid)->elements == NULL) {
	free(*dest_oid);
	*dest_oid = GSS_C_NO_OID;
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }
    memcpy((*dest_oid)->elements, src_oid->elements, src_oid->length);
    (*dest_oid)->length = src_oid->length;

    *minor_status = 0;
    return GSS_S_COMPLETE;
}
