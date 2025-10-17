/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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

#include "combining_table.h"

static int
translation_cmp(const void *key, const void *data)
{
    const struct translation *t1 = (const struct translation *)key;
    const struct translation *t2 = (const struct translation *)data;

    return t1->key - t2->key;
}

int
_wind_combining_class(uint32_t code_point)
{
    struct translation ts = {code_point};
    void *s = bsearch(&ts, _wind_combining_table, _wind_combining_table_size,
		      sizeof(_wind_combining_table[0]),
		      translation_cmp);
    if (s != NULL) {
	const struct translation *t = (const struct translation *)s;
	return t->combining_class;
    } else {
	return 0;
    }
}
