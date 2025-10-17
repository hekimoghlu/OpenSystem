/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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

void
der_free_general_string (heim_general_string *str)
{
    free(*str);
    *str = NULL;
}

void
der_free_integer (int *i)
{
    *i = 0;
}

void
der_free_unsigned (unsigned *u)
{
    *u = 0;
}

void
der_free_generalized_time(time_t *t)
{
    *t = 0;
}

void
der_free_utctime(time_t *t)
{
    *t = 0;
}


void
der_free_utf8string (heim_utf8_string *str)
{
    free(*str);
    *str = NULL;
}

void
der_free_printable_string (heim_printable_string *str)
{
    der_free_octet_string(str);
}

void
der_free_ia5_string (heim_ia5_string *str)
{
    der_free_octet_string(str);
}

void
der_free_bmp_string (heim_bmp_string *k)
{
    free(k->data);
    k->data = NULL;
    k->length = 0;
}

void
der_free_universal_string (heim_universal_string *k)
{
    free(k->data);
    k->data = NULL;
    k->length = 0;
}

void
der_free_visible_string (heim_visible_string *str)
{
    free(*str);
    *str = NULL;
}

void
der_free_octet_string (heim_octet_string *k)
{
    free(k->data);
    k->data = NULL;
    k->length = 0;
}

void
der_free_heim_integer (heim_integer *k)
{
    free(k->data);
    k->data = NULL;
    k->length = 0;
}

void
der_free_oid (heim_oid *k)
{
    free(k->components);
    k->components = NULL;
    k->length = 0;
}

void
der_free_bit_string (heim_bit_string *k)
{
    free(k->data);
    k->data = NULL;
    k->length = 0;
}
