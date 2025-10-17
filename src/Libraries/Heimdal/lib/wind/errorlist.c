/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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

#include "errorlist_table.h"

static int
error_entry_cmp(const void *a, const void *b)
{
    const struct error_entry *ea = (const struct error_entry*)a;
    const struct error_entry *eb = (const struct error_entry*)b;

    if (ea->start >= eb->start && ea->start < eb->start + eb->len)
	return 0;
    return ea->start - eb->start;
}

int
_wind_stringprep_error(const uint32_t cp, wind_profile_flags flags)
{
    struct error_entry ee = {cp};
    const struct error_entry *s;

    s = (const struct error_entry *)
	bsearch(&ee, _wind_errorlist_table,
		_wind_errorlist_table_size,
		sizeof(_wind_errorlist_table[0]),
		error_entry_cmp);
    if (s == NULL)
	return 0;
    return (s->flags & flags);
}

int
_wind_stringprep_prohibited(const uint32_t *in, size_t in_len,
			    wind_profile_flags flags)
{
    unsigned i;

    for (i = 0; i < in_len; ++i)
	if (_wind_stringprep_error(in[i], flags))
	    return 1;
    return 0;
}
