/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#include "der_locl.h"
#include "heim_asn1.h"

RCSID("$Id$");

int
encode_heim_any(unsigned char *p, size_t len,
		const heim_any *data, size_t *size)
{
    return der_put_octet_string (p, len, data, size);
}

int
decode_heim_any(const unsigned char *p, size_t len,
		heim_any *data, size_t *size)
{
    size_t len_len, length, l;
    Der_class thisclass;
    Der_type thistype;
    unsigned int thistag;
    int e;

    memset(data, 0, sizeof(*data));

    e = der_get_tag (p, len, &thisclass, &thistype, &thistag, &l);
    if (e) return e;
    if (l > len)
	return ASN1_OVERFLOW;
    e = der_get_length(p + l, len - l, &length, &len_len);
    if (e) return e;
    if (length == ASN1_INDEFINITE) {
        if (len < len_len + l)
	    return ASN1_OVERFLOW;
	length = len - (len_len + l);
    } else {
	if (len < length + len_len + l)
	    return ASN1_OVERFLOW;
    }

    data->data = malloc(length + len_len + l);
    if (data->data == NULL)
	return ENOMEM;
    data->length = length + len_len + l;
    memcpy(data->data, p, length + len_len + l);

    if (size)
	*size = length + len_len + l;

    return 0;
}

void
free_heim_any(heim_any *data)
{
    der_free_octet_string(data);
}

size_t
length_heim_any(const heim_any *data)
{
    return data->length;
}

int
copy_heim_any(const heim_any *from, heim_any *to)
{
    return der_copy_octet_string(from, to);
}

int
encode_heim_any_set(unsigned char *p, size_t len,
		    const heim_any_set *data, size_t *size)
{
    return der_put_octet_string (p, len, data, size);
}

int
decode_heim_any_set(const unsigned char *p, size_t len,
		heim_any_set *data, size_t *size)
{
    return der_get_octet_string(p, len, data, size);
}

void
free_heim_any_set(heim_any_set *data)
{
    der_free_octet_string(data);
}

size_t
length_heim_any_set(const heim_any *data)
{
    return data->length;
}

int
copy_heim_any_set(const heim_any_set *from, heim_any_set *to)
{
    return der_copy_octet_string(from, to);
}

int
heim_any_cmp(const heim_any_set *p, const heim_any_set *q)
{
    return der_heim_octet_string_cmp(p, q);
}
