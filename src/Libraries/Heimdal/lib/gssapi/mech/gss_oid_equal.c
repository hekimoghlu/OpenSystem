/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
 * Compare two GSS-API OIDs with each other.
 *
 * GSS_C_NO_OID matches nothing, not even it-self.
 *
 * @param a first oid to compare
 * @param b second oid to compare
 *
 * @return non-zero when both oid are the same OID, zero when they are
 *         not the same.
 *
 * @ingroup gssapi
 */

GSSAPI_LIB_FUNCTION int GSSAPI_LIB_CALL
gss_oid_equal(__nullable gss_const_OID a, __nullable gss_const_OID b)
{
    if (a == b && a != GSS_C_NO_OID)
	return 1;
    if (a == GSS_C_NO_OID || b == GSS_C_NO_OID || a->length != b->length)
	return 0;
    return memcmp(a->elements, b->elements, a->length) == 0;
}
