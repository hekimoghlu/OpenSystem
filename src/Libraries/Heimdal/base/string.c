/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include <string.h>

static void
string_dealloc(void *ptr)
{
}

static int
string_cmp(void *a, void *b)
{
    return strcmp(a, b);
}

static unsigned long
string_hash(void *ptr)
{
    const char *s = ptr;
    unsigned long n;

    for (n = 0; *s; ++s)
	n += *s;
    return n;
}

struct heim_type_data _heim_string_object = {
    HEIM_TID_STRING,
    "string-object",
    NULL,
    string_dealloc,
    NULL,
    string_cmp,
    string_hash
};

/**
 * Create a string object
 *
 * @param string the string to create, must be an utf8 string
 *
 * @return string object
 */

heim_string_t
heim_string_create(const char *string)
{
    size_t len = strlen(string);
    heim_string_t s;

    s = _heim_alloc_object(&_heim_string_object, len + 1);
    if (s)
	memcpy(s, string, len + 1);
    return s;
}

/**
 * Return the type ID of string objects
 *
 * @return type id of string objects
 */

heim_tid_t
heim_string_get_type_id(void)
{
    return HEIM_TID_STRING;
}

/**
 * Get the string value of the content.
 *
 * @param string the string object to get the value from
 *
 * @return a utf8 string
 */

char *
heim_string_copy_utf8(heim_string_t string)
{
    return strdup(string);
}
