/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
#include "baselocl.h"

static void
number_dealloc(void *ptr)
{
}

static int
number_cmp(void *a, void *b)
{
    int na, nb;

    if (heim_base_is_tagged_object(a))
	na = heim_base_tagged_object_value(a);
    else
	na = *(int *)a;

    if (heim_base_is_tagged_object(b))
	nb = heim_base_tagged_object_value(b);
    else
	nb = *(int *)b;

    return na - nb;
}

static unsigned long
number_hash(void *ptr)
{
    if (heim_base_is_tagged_object(ptr))
	return heim_base_tagged_object_value(ptr);
    return (unsigned long)*(int *)ptr;
}

struct heim_type_data _heim_number_object = {
    HEIM_TID_NUMBER,
    "number-object",
    NULL,
    number_dealloc,
    NULL,
    number_cmp,
    number_hash
};

/**
 * Create a number object
 *
 * @param the number to contain in the object
 *
 * @return a number object
 */

heim_number_t
heim_number_create(int number)
{
    heim_number_t n;

    if (number < 0xffffff && number >= 0)
	return heim_base_make_tagged_object(number, HEIM_TID_NUMBER);

    n = _heim_alloc_object(&_heim_number_object, sizeof(int));
    if (n)
	*((int *)n) = number;
    return n;
}

/**
 * Return the type ID of number objects
 *
 * @return type id of number objects
 */

heim_tid_t
heim_number_get_type_id(void)
{
    return HEIM_TID_NUMBER;
}

/**
 * Get the int value of the content
 *
 * @param number the number object to get the value from
 *
 * @return an int
 */

int
heim_number_get_int(heim_number_t number)
{
    if (heim_base_is_tagged_object(number))
	return (int)heim_base_tagged_object_value(number);
    return *(int *)number;
}
