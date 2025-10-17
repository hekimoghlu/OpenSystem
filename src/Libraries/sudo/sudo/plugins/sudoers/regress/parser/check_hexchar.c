/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SUDO_ERROR_WRAP 0

#include "sudo_compat.h"
#include "sudo_util.h"

int hexchar(const char *s);

sudo_dso_public int main(int argc, char *argv[]);

struct hexchar_test {
    char hex[3];
    int value;
};

int
main(int argc, char *argv[])
{
    struct hexchar_test *test_data;
    int i, ntests, result, errors = 0;
    static const char xdigs_lower[] = "0123456789abcdef";
    static const char xdigs_upper[] = "0123456789ABCDEF";

    initprogname(argc > 0 ? argv[0] : "check_hexchar");

    /* Build up test data. */
    ntests = 256 + 256 + 3;
    test_data = calloc(sizeof(*test_data), ntests);
    for (i = 0; i < 256; i++) {
	/* lower case */
	test_data[i].value = i;
	test_data[i].hex[1] = xdigs_lower[ (i & 0x0f)];
	test_data[i].hex[0] = xdigs_lower[((i & 0xf0) >> 4)];
	/* upper case */
	test_data[i + 256].value = i;
	test_data[i + 256].hex[1] = xdigs_upper[ (i & 0x0f)];
	test_data[i + 256].hex[0] = xdigs_upper[((i & 0xf0) >> 4)];
    }
    /* Also test invalid data */
    test_data[ntests - 3].hex[0] = '\0';
    test_data[ntests - 3].value = -1;
    strlcpy(test_data[ntests - 2].hex, "AG", sizeof(test_data[ntests - 2].hex));
    test_data[ntests - 2].value = -1;
    strlcpy(test_data[ntests - 1].hex, "-1", sizeof(test_data[ntests - 1].hex));
    test_data[ntests - 1].value = -1;

    for (i = 0; i < ntests; i++) {
	result = hexchar(test_data[i].hex);
	if (result != test_data[i].value) {
	    fprintf(stderr, "check_hexchar: expected %d, got %d\n",
		test_data[i].value, result);
	    errors++;
	}
    }
    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }
    exit(errors);
}
