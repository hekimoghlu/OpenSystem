/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
gss_encapsulate_token(__nonnull gss_const_buffer_t input_token,
		      __nonnull gss_const_OID oid,
		      __nonnull gss_buffer_t output_token)
{
    GSSAPIContextToken ct;
    int ret;
    size_t size;

    ret = der_get_oid (oid->elements, oid->length, &ct.thisMech, &size);
    if (ret) {
	_mg_buffer_zero(output_token);
	return GSS_S_FAILURE;
    }

    ct.innerContextToken.data = input_token->value;
    ct.innerContextToken.length = input_token->length;

    ASN1_MALLOC_ENCODE(GSSAPIContextToken,
		       output_token->value, output_token->length,
		       &ct, &size, ret);
    der_free_oid(&ct.thisMech);
    if (ret) {
	_mg_buffer_zero(output_token);
	return GSS_S_FAILURE;
    }
    if (output_token->length != size)
	abort();

    return GSS_S_COMPLETE;
}
