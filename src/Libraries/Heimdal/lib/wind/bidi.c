/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#include "windlocl.h"

#include <stdlib.h>

#include "bidi_table.h"

static int
range_entry_cmp(const void *a, const void *b)
{
    const struct range_entry *ea = (const struct range_entry*)a;
    const struct range_entry *eb = (const struct range_entry*)b;

    if (ea->start >= eb->start && ea->start < eb->start + eb->len)
	return 0;
    return ea->start - eb->start;
}

static int
is_ral(uint32_t cp)
{
    struct range_entry ee = {cp};
    void *s = bsearch(&ee, _wind_ral_table, _wind_ral_table_size,
		      sizeof(_wind_ral_table[0]),
		      range_entry_cmp);
    return s != NULL;
}

static int
is_l(uint32_t cp)
{
    struct range_entry ee = {cp};
    void *s = bsearch(&ee, _wind_l_table, _wind_l_table_size,
		      sizeof(_wind_l_table[0]),
		      range_entry_cmp);
    return s != NULL;
}

int
_wind_stringprep_testbidi(const uint32_t *in, size_t in_len, wind_profile_flags flags)
{
    size_t i;
    unsigned ral = 0;
    unsigned l   = 0;

    if ((flags & (WIND_PROFILE_NAME|WIND_PROFILE_SASL)) == 0)
	return 0;

    for (i = 0; i < in_len; ++i) {
	ral |= is_ral(in[i]);
	l   |= is_l(in[i]);
    }
    if (ral) {
	if (l)
	    return 1;
	if (!is_ral(in[0]) || !is_ral(in[in_len - 1]))
	    return 1;
    }
    return 0;
}
