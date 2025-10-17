/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 27, 2025.
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
#include <string.h>

#include "windlocl.h"
#include "punycode_examples.h"

int
main(void)
{
    unsigned i;
    unsigned failures = 0;

    for (i = 0; i < punycode_examples_size; ++i) {
	char buf[256];
	int ret;
	const struct punycode_example *e = &punycode_examples[i];
	size_t len;

	len = sizeof(buf);
	ret = wind_punycode_label_toascii(e->val, e->len, buf, &len);
	if (ret) {
	    printf("punycode %u (%s) failed: %d\n", i, e->description, ret);
	    ++failures;
	    continue;
	}
	if (strncmp(buf, "xn--", 4) == 0) {
	    memmove(buf, buf + 4, len - 4);
	    len -= 4;
	}
	if (len != strlen(e->pc)) {
	    printf("punycode %u (%s) wrong len, actual: %u, expected: %u\n",
		   i, e->description,
		   (unsigned int)len, (unsigned int)strlen(e->pc));
	    printf("buf %s != pc: %s\n", buf, e->pc);
	    ++failures;
	    continue;
	}
	if (strncasecmp(buf, e->pc, len) != 0) {
	    printf("punycode %u (%s) wrong contents, "
		   "actual: \"%.*s\", expected: \"%s\"\n",
		   i, e->description, (unsigned int)len, buf, e->pc);
	    ++failures;
	    continue;
	}
    }
    return failures != 0;
}
