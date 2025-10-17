/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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

int
der_heim_oid_cmp(const heim_oid *p, const heim_oid *q)
{
    if (p->length != q->length)
	return (int)(p->length - q->length);
    return memcmp(p->components,
		  q->components,
		  p->length * sizeof(*p->components));
}

int
der_heim_octet_string_cmp(const heim_octet_string *p,
			  const heim_octet_string *q)
{
    if (p->length != q->length)
	return (int)(p->length - q->length);
    return memcmp(p->data, q->data, p->length);
}

int
der_printable_string_cmp(const heim_printable_string *p,
			 const heim_printable_string *q)
{
    return der_heim_octet_string_cmp(p, q);
}

int
der_ia5_string_cmp(const heim_ia5_string *p,
		   const heim_ia5_string *q)
{
    return der_heim_octet_string_cmp(p, q);
}

int
der_heim_bit_string_cmp(const heim_bit_string *p,
			const heim_bit_string *q)
{
    int r1, r2;
    size_t i;
    if (p->length != q->length)
	return (int)(p->length - q->length);
    i = memcmp(p->data, q->data, p->length / 8);
    if (i)
	return (int)i;
    if ((p->length % 8) == 0)
	return 0;
    i = (p->length / 8);
    r1 = ((unsigned char *)p->data)[i];
    r2 = ((unsigned char *)q->data)[i];
    i = 8 - (p->length % 8);
    r1 = r1 >> i;
    r2 = r2 >> i;
    return r1 - r2;
}

int
der_heim_integer_cmp(const heim_integer *p,
		     const heim_integer *q)
{
    if (p->negative != q->negative)
	return q->negative - p->negative;
    if (p->length != q->length)
	return (int)(p->length - q->length);
    return memcmp(p->data, q->data, p->length);
}

int
der_heim_bmp_string_cmp(const heim_bmp_string *p, const heim_bmp_string *q)
{
    if (p->length != q->length)
	return (int)(p->length - q->length);
    return memcmp(p->data, q->data, q->length * sizeof(q->data[0]));
}

int
der_heim_universal_string_cmp(const heim_universal_string *p,
			      const heim_universal_string *q)
{
    if (p->length != q->length)
	return (int)(p->length - q->length);
    return memcmp(p->data, q->data, q->length * sizeof(q->data[0]));
}
