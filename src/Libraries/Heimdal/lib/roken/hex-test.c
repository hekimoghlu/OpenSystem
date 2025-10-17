/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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

#include "roken.h"
#include <hex.h>

int
main(int argc, char **argv)
{
    int numerr = 0;
    int numtest = 1;
    struct test {
	void *data;
	size_t len;
	const char *result;
    } *t, tests[] = {
	{ "", 0 , "" },
	{ "a", 1, "61" },
	{ "ab", 2, "6162" },
	{ "abc", 3, "616263" },
	{ "abcd", 4, "61626364" },
	{ "abcde", 5, "6162636465" },
	{ "abcdef", 6, "616263646566" },
	{ "abcdefg", 7, "61626364656667" },
	{ "=", 1, "3D" },
	{ NULL }
    };
    for(t = tests; t->data; t++) {
	char *str;
	int len;
	len = hex_encode(t->data, t->len, &str);
	if(strcmp(str, t->result) != 0) {
	    fprintf(stderr, "failed test %d: %s != %s\n", numtest,
		    str, t->result);
	    numerr++;
	}
	free(str);
	str = strdup(t->result);
	len = strlen(str);
	len = hex_decode(t->result, str, len);
	if(len != t->len) {
	    fprintf(stderr, "failed test %d: len %lu != %lu\n", numtest,
		    (unsigned long)len, (unsigned long)t->len);
	    numerr++;
	} else if(memcmp(str, t->data, t->len) != 0) {
	    fprintf(stderr, "failed test %d: data\n", numtest);
	    numerr++;
	}
	free(str);
	numtest++;
    }

    {
	unsigned char buf[2] = { 0, 0xff } ;
	int len;

	len = hex_decode("A", buf, 1);
	if (len != 1) {
	    fprintf(stderr, "len != 1");
	    numerr++;
	}
	if (buf[0] != 10) {
	    fprintf(stderr, "buf != 10");
	    numerr++;
	}
	if (buf[1] != 0xff) {
	    fprintf(stderr, "buf != 0xff");
	    numerr++;
	}

    }

    return numerr;
}
