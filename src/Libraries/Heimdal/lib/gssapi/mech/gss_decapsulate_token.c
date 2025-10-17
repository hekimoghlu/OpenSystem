/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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
gss_decapsulate_token(__nonnull gss_const_buffer_t input_token,
		      __nonnull gss_const_OID oid,
		      __nonnull gss_buffer_t output_token)
{
    GSSAPIContextToken ct;
    heim_oid o;
    OM_uint32 status;
    int ret;
    size_t size;

    _mg_buffer_zero(output_token);

    ret = der_get_oid (oid->elements, oid->length, &o, &size);
    if (ret)
	return GSS_S_FAILURE;

    ret = decode_GSSAPIContextToken(input_token->value, input_token->length,
				    &ct, NULL);
    if (ret) {
	der_free_oid(&o);
	return GSS_S_FAILURE;
    }

    if (der_heim_oid_cmp(&ct.thisMech, &o) == 0) {
	status = GSS_S_COMPLETE;
	output_token->value = ct.innerContextToken.data;
	output_token->length = ct.innerContextToken.length;
	der_free_oid(&ct.thisMech);
    } else {
	free_GSSAPIContextToken(&ct);
 	status = GSS_S_FAILURE;
    }
    der_free_oid(&o);

    return status;
}
