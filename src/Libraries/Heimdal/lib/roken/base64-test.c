/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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
#include <base64.h>

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
	{ "1", 1, "MQ==" },
	{ "22", 2, "MjI=" },
	{ "333", 3, "MzMz" },
	{ "4444", 4, "NDQ0NA==" },
	{ "55555", 5, "NTU1NTU=" },
	{ "abc:def", 7, "YWJjOmRlZg==" },
	{ NULL }
    };
    for(t = tests; t->data; t++) {
	char *str;
	int len;
	len = base64_encode(t->data, t->len, &str);
	if(strcmp(str, t->result) != 0) {
	    fprintf(stderr, "failed test %d: %s != %s\n", numtest,
		    str, t->result);
	    numerr++;
	}
	free(str);
	str = strdup(t->result);
	len = base64_decode(t->result, str);
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
	char str[32];
	if(base64_decode("M=M=", str) != -1) {
	    fprintf(stderr, "failed test %d: successful decode of `M=M='\n",
		    numtest++);
	    numerr++;
	}
	if(base64_decode("MQ===", str) != -1) {
	    fprintf(stderr, "failed test %d: successful decode of `MQ==='\n",
		    numtest++);
	    numerr++;
	}
    }
    return numerr;
}
