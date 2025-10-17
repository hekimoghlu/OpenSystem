/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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

RCSID("$Id$");

int
der_copy_general_string (const heim_general_string *from,
			 heim_general_string *to)
{
    *to = strdup(*from);
    if(*to == NULL)
	return ENOMEM;
    return 0;
}

int
der_copy_integer (const int *from, int *to)
{
    *to = *from;
    return 0;
}

int
der_copy_unsigned (const unsigned *from, unsigned *to)
{
    *to = *from;
    return 0;
}

int
der_copy_generalized_time (const time_t *from, time_t *to)
{
    *to = *from;
    return 0;
}

int
der_copy_utctime (const time_t *from, time_t *to)
{
    *to = *from;
    return 0;
}

int
der_copy_utf8string (const heim_utf8_string *from, heim_utf8_string *to)
{
    return der_copy_general_string(from, to);
}

int
der_copy_printable_string (const heim_printable_string *from,
		       heim_printable_string *to)
{
    to->length = from->length;
    to->data   = malloc(to->length + 1);
    if(to->data == NULL)
	return ENOMEM;
    memcpy(to->data, from->data, to->length);
    ((char *)to->data)[to->length] = '\0';
    return 0;
}

int
der_copy_ia5_string (const heim_ia5_string *from,
		     heim_ia5_string *to)
{
    return der_copy_printable_string(from, to);
}

int
der_copy_bmp_string (const heim_bmp_string *from, heim_bmp_string *to)
{
    to->length = from->length;
    to->data   = malloc(to->length * sizeof(to->data[0]));
    if(to->length != 0 && to->data == NULL)
	return ENOMEM;
    memcpy(to->data, from->data, to->length * sizeof(to->data[0]));
    return 0;
}

int
der_copy_universal_string (const heim_universal_string *from,
			   heim_universal_string *to)
{
    to->length = from->length;
    to->data   = malloc(to->length * sizeof(to->data[0]));
    if(to->length != 0 && to->data == NULL)
	return ENOMEM;
    memcpy(to->data, from->data, to->length * sizeof(to->data[0]));
    return 0;
}

int
der_copy_visible_string (const heim_visible_string *from,
			 heim_visible_string *to)
{
    return der_copy_general_string(from, to);
}

int
der_copy_octet_string (const heim_octet_string *from, heim_octet_string *to)
{
    to->length = from->length;
    to->data   = malloc(to->length);
    if(to->length != 0 && to->data == NULL)
	return ENOMEM;
    memcpy(to->data, from->data, to->length);
    return 0;
}

int
der_copy_heim_integer (const heim_integer *from, heim_integer *to)
{
    to->length = from->length;
    to->data   = malloc(to->length);
    if(to->length != 0 && to->data == NULL)
	return ENOMEM;
    memcpy(to->data, from->data, to->length);
    to->negative = from->negative;
    return 0;
}

int
der_copy_oid (const heim_oid *from, heim_oid *to)
{
    to->length     = from->length;
    to->components = malloc(to->length * sizeof(*to->components));
    if (to->length != 0 && to->components == NULL)
	return ENOMEM;
    memcpy(to->components, from->components,
	   to->length * sizeof(*to->components));
    return 0;
}

int
der_copy_bit_string (const heim_bit_string *from, heim_bit_string *to)
{
    size_t len;

    len = (from->length + 7) / 8;
    to->length = from->length;
    to->data   = malloc(len);
    if(len != 0 && to->data == NULL)
	return ENOMEM;
    memcpy(to->data, from->data, len);
    return 0;
}
