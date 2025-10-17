/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 25, 2024.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include "windlocl.h"

#include <stdlib.h>

#include "map_table.h"

static int
translation_cmp(const void *key, const void *data)
{
    const struct translation *t1 = (const struct translation *)key;
    const struct translation *t2 = (const struct translation *)data;

    return t1->key - t2->key;
}

int
_wind_stringprep_map(const uint32_t *in, size_t in_len,
		     uint32_t *out, size_t *out_len,
		     wind_profile_flags flags)
{
    unsigned i;
    unsigned o = 0;

    for (i = 0; i < in_len; ++i) {
	struct translation ts = {in[i]};
	const struct translation *s;

	s = (const struct translation *)
	    bsearch(&ts, _wind_map_table, _wind_map_table_size,
		    sizeof(_wind_map_table[0]),
		    translation_cmp);
	if (s != NULL && (s->flags & flags)) {
	    unsigned j;

	    for (j = 0; j < s->val_len; ++j) {
		if (o >= *out_len)
		    return WIND_ERR_OVERRUN;
		out[o++] = _wind_map_table_val[s->val_offset + j];
	    }
	} else {
	    if (o >= *out_len)
		return WIND_ERR_OVERRUN;
	    out[o++] = in[i];

	}
    }
    *out_len = o;
    return 0;
}
