/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#include "parse_time.h"

static struct units time_units[] = {
    {"year",	365 * 24 * 60 * 60},
    {"month",	30 * 24 * 60 * 60},
    {"week",	7 * 24 * 60 * 60},
    {"day",	24 * 60 * 60},
    {"hour",	60 * 60},
    {"h",	60 * 60},
    {"minute",	60},
    {"m",	60},
    {"second",	1},
    {"s",	1},
    {NULL, 0},
};

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
parse_time (const char *s, const char *def_unit)
{
    return parse_units (s, time_units, def_unit);
}

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
unparse_time (int t, char *s, size_t len)
{
    return unparse_units (t, time_units, s, len);
}

ROKEN_LIB_FUNCTION size_t ROKEN_LIB_CALL
unparse_time_approx (int t, char *s, size_t len)
{
    return unparse_units_approx (t, time_units, s, len);
}

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
print_time_table (FILE *f)
{
    print_units_table (time_units, f);
}
