/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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
#include <config.h>

#include <parse_units.h>
#include "parse_bytes.h"

static struct units bytes_units[] = {
    { "gigabyte", 1024 * 1024 * 1024 },
    { "gbyte", 1024 * 1024 * 1024 },
    { "GB", 1024 * 1024 * 1024 },
    { "megabyte", 1024 * 1024 },
    { "mbyte", 1024 * 1024 },
    { "MB", 1024 * 1024 },
    { "kilobyte", 1024 },
    { "KB", 1024 },
    { "byte", 1 },
    { NULL, 0 }
};

static struct units bytes_short_units[] = {
    { "GB", 1024 * 1024 * 1024 },
    { "MB", 1024 * 1024 },
    { "KB", 1024 },
    { NULL, 0 }
};

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
parse_bytes (const char *s, const char *def_unit)
{
    return parse_units (s, bytes_units, def_unit);
}

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
unparse_bytes (int t, char *s, size_t len)
{
    return unparse_units (t, bytes_units, s, len);
}

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
unparse_bytes_short (int t, char *s, size_t len)
{
    return unparse_units_approx (t, bytes_short_units, s, len);
}
