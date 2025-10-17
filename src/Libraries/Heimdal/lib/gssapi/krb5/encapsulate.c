/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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

void
_gssapi_encap_length (size_t data_len,
		      size_t *len,
		      size_t *total_len,
		      const gss_OID mech)
{
    size_t len_len;

    *len = 1 + 1 + mech->length + data_len;

    len_len = der_length_len(*len);

    *total_len = 1 + len_len + *len;
}

void
_gsskrb5_encap_length (size_t data_len,
			  size_t *len,
			  size_t *total_len,
			  const gss_OID mech)
{
    _gssapi_encap_length(data_len + 2, len, total_len, mech);
}

void *
_gsskrb5_make_header (void *ptr,
			 size_t len,
			 const void *type,
			 const gss_OID mech)
{
    u_char *p = ptr;
    p = _gssapi_make_mech_header(p, len, mech);
    memcpy (p, type, 2);
    p += 2;
    return p;
}

void *
_gssapi_make_mech_header(void *ptr,
			 size_t len,
			 const gss_OID mech)
{
    u_char *p = ptr;
    int e;
    size_t len_len, foo;

    *p++ = 0x60;
    len_len = der_length_len(len);
    e = der_put_length (p + len_len - 1, len_len, len, &foo);
    if(e || foo != len_len)
	abort ();
    p += len_len;
    *p++ = 0x06;
    *p++ = mech->length;
    memcpy (p, mech->elements, mech->length);
    p += mech->length;
    return p;
}

/*
 * Give it a krb5_data and it will encapsulate with extra GSS-API wrappings.
 */

OM_uint32
_gssapi_encapsulate(
    OM_uint32 *minor_status,
    const krb5_data *in_data,
    gss_buffer_t output_token,
    const gss_OID mech
)
{
    size_t len, outer_len;
    void *p;

    _gssapi_encap_length (in_data->length, &len, &outer_len, mech);

    output_token->length = outer_len;
    output_token->value  = malloc (outer_len);
    if (output_token->value == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    p = _gssapi_make_mech_header (output_token->value, len, mech);
    memcpy (p, in_data->data, in_data->length);
    return GSS_S_COMPLETE;
}

/*
 * Give it a krb5_data and it will encapsulate with extra GSS-API krb5
 * wrappings.
 */

OM_uint32
_gsskrb5_encapsulate(
			OM_uint32 *minor_status,
			const krb5_data *in_data,
			gss_buffer_t output_token,
			const void *type,
			const gss_OID mech
)
{
    size_t len, outer_len;
    u_char *p;

    _gsskrb5_encap_length (in_data->length, &len, &outer_len, mech);

    output_token->length = outer_len;
    output_token->value  = malloc (outer_len);
    if (output_token->value == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    p = _gsskrb5_make_header (output_token->value, len, type, mech);
    memcpy (p, in_data->data, in_data->length);
    return GSS_S_COMPLETE;
}
