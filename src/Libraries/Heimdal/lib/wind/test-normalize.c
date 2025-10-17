/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

#include <roken.h>

#include "windlocl.h"
#include "normalize_table.h"

static size_t
parse_vector(char *buf, uint32_t *v)
{
    char *last;
    unsigned ret = 0;
    const char *n;
    unsigned u;

    for(n = strtok_r(buf, " ", &last);
	n != NULL;
	n = strtok_r(NULL, " ", &last)) {
	if (ret >= MAX_LENGTH_CANON) {
	    errx(1, "increase MAX_LENGTH_CANON");
	}
	if (sscanf(n, "%x", &u) != 1) {
	    errx(1, "failed to parse hex: %s", n);
	}
	v[ret] = u;
	++ret;
    }
    return ret;
}

static void
dump_vector(const char * msg, uint32_t * v, size_t len)
{
    size_t i;

    printf("%s: (%d) ", msg, (int)len);
    for (i=0; i < len; i++) {
	printf("%s%x", (i > 0? " ":""), v[i]);
    }
    printf("\n");
}

static int
test(char *buf, unsigned lineno)
{
    char *last;
    char *c;
    uint32_t in[MAX_LENGTH_CANON];
    size_t in_len;
    uint32_t out[MAX_LENGTH_CANON];
    size_t out_len;
    uint32_t *tmp;
    size_t norm_len;
    int ret;

    c = strtok_r(buf, ";", &last);
    if (c == NULL)
	return 0;

    in_len = parse_vector(c, in);
    if (strtok_r(NULL, ";", &last) == NULL)
	return 0;
    if (strtok_r(NULL, ";", &last) == NULL)
	return 0;
    c = strtok_r(NULL, ";", &last);
    if (c == NULL)
	return 0;
    out_len = parse_vector(c, out);
    if (strtok_r(NULL, ";", &last) == NULL)
	return 0;
    c = last;

    norm_len = MAX_LENGTH_CANON;
    tmp = malloc(norm_len * sizeof(uint32_t));
    if (tmp == NULL && norm_len != 0)
	err(1, "malloc");
    ret = _wind_stringprep_normalize(in, in_len, tmp, &norm_len);
    if (ret) {
	printf("wind_stringprep_normalize %s failed\n", c);
	free(tmp);
	return 1;
    }
    if (out_len != norm_len) {
	printf("%u: wrong out len (%s)\n", lineno, c);
	dump_vector("Expected", out, out_len);
	dump_vector("Received", tmp, norm_len);
	free(tmp);
	return 1;
    }
    if (memcmp(out, tmp, out_len * sizeof(uint32_t)) != 0) {
	printf("%u: wrong out data (%s)\n", lineno, c);
	dump_vector("Expected", out, out_len);
	dump_vector("Received", tmp, norm_len);
	free(tmp);
	return 1;
    }
    free(tmp);
    return 0;
}

int
main(int argc, char **argv)
{
    FILE *f;
    char buf[1024];
    char filename[256] = "NormalizationTest.txt";
    unsigned failures = 0;
    unsigned lineno = 0;

    if (argc > 2)
	errx(1, "usage: %s [file]", argv[0]);
    else if (argc == 2)
	strlcpy(filename, argv[1], sizeof(filename));

    f = fopen(filename, "r");
    if (f == NULL) {
	const char *srcdir = getenv("srcdir");
	if (srcdir != NULL) {
	    char longname[256];
	    snprintf(longname, sizeof(longname), "%s/%s", srcdir, filename);
	    f = fopen(longname, "r");
	}
	if (f == NULL)
	    err(1, "open %s", filename);
    }
    while (fgets(buf, sizeof(buf), f) != NULL) {
	lineno++;
	if (buf[0] == '#')
	    continue;
	if (buf[0] == '@') {
	    continue;
	}
	failures += test(buf, lineno);
    }
    fclose(f);
    return failures != 0;
}
