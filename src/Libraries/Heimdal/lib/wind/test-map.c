/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

#include "windlocl.h"

#define MAX_LENGTH 2

struct example {
    uint32_t in[MAX_LENGTH];
    size_t in_len;
    uint32_t out[MAX_LENGTH];
    size_t out_len;
};

static struct example cases[] = {
    {{0}, 0, {0}, 0},
    {{0x0041}, 1, {0x0061}, 1},
    {{0x0061}, 1, {0x0061}, 1},
    {{0x00AD}, 1, {0}, 0},
    {{0x00DF}, 1, {0x0073, 0x0073}, 2}
};

static int
try(const struct example *c)
{
    int ret;
    size_t out_len = c->out_len;
    uint32_t *tmp = malloc(out_len * sizeof(uint32_t));
    if (tmp == NULL && out_len != 0)
	err(1, "malloc");
    ret = _wind_stringprep_map(c->in, c->in_len, tmp, &out_len, WIND_PROFILE_NAME);
    if (ret) {
	printf("wind_stringprep_map failed\n");
	return 1;
    }
    if (out_len != c->out_len) {
	printf("wrong out len\n");
	free(tmp);
	return 1;
    }
    if (memcmp(c->out, tmp, out_len * sizeof(uint32_t)) != 0) {
	printf("wrong out data\n");
	free(tmp);
	return 1;
    }
    free(tmp);
    return 0;
}

int
main(void)
{
    unsigned i;
    unsigned failures = 0;

    for (i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i)
	failures += try(&cases[i]);
    return failures != 0;
}

