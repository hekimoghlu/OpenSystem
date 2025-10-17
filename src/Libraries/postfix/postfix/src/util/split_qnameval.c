/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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
/* System libraries. */

#include <sys_defs.h>
#include <ctype.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <stringops.h>

/* split_qnameval - split "key = value", support quoted key */

const char *split_qnameval(char *buf, char **pkey, char **pvalue)
{
    int     in_quotes = 0;
    char   *key;
    char   *key_end;
    char   *value;

    for (key = buf; *key && ISSPACE(*key); key++)
	 /* void */ ;
    if (*key == 0)
	return ("no key found; expected format: key = value");

    for (key_end = key; *key_end; key_end++) {
	if (*key_end == '\\') {
	    if (*++key_end == 0)
		break;
	} else if (ISSPACE(*key_end) || *key_end == '=') {
	    if (!in_quotes)
		break;
	} else if (*key_end == '"') {
	    in_quotes = !in_quotes;
	}
    }
    if (in_quotes) {
	return ("unbalanced '\"\'");
    }
    value = key_end;
    while (ISSPACE(*value))
	value++;
    if (*value != '=')
	return ("missing '=' after attribute name");
    *key_end = 0;
    do {
	value++;
    } while (ISSPACE(*value));
    trimblanks(value, 0)[0] = 0;
    *pkey = key;
    *pvalue = value;
    return (0);
}

#ifdef TEST

#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <mymalloc.h>

static int compare(int test_number, const char *what,
		           const char *expect, const char *real)
{
    if ((expect == 0 && real == 0)
	|| (expect != 0 && real != 0 && strcmp(expect, real) == 0)) {
	return (0);
    } else {
	msg_warn("test %d: %s mis-match: expect='%s', real='%s'",
		 test_number, what, expect ? expect : "(null)",
		 real ? real : "(null)");
	return (1);
    }
}

int     main(int argc, char **argv)
{
    struct test_info {
	const char *input;
	const char *expect_result;
	const char *expect_key;
	const char *expect_value;
    };
    static const struct test_info test_info[] = {
	/* Unquoted keys. */
	{"xx = yy", 0, "xx", "yy"},
	{"xx=yy", 0, "xx", "yy"},
	{"xx =", 0, "xx", ""},
	{"xx=", 0, "xx", ""},
	{"xx", "missing '=' after attribute name", 0, 0},
	/* Quoted keys. */
	{"\"xx \" = yy", 0, "\"xx \"", "yy"},
	{"\"xx \"= yy", 0, "\"xx \"", "yy"},
	{"\"xx \" =", 0, "\"xx \"", ""},
	{"\"xx \"=", 0, "\"xx \"", ""},
	{"\"xx \"", "missing '=' after attribute name", 0, 0},
	{"\"xx ", "unbalanced '\"'", 0, 0},
	/* Backslash escapes. */
	{"\"\\\"xx \" = yy", 0, "\"\\\"xx \"", "yy"},
	{0,},
    };

    int     errs = 0;
    const struct test_info *tp;

    for (tp = test_info; tp->input != 0; tp++) {
	const char *result;
	char   *key = 0;
	char   *value = 0;
	char   *buf = mystrdup(tp->input);
	int     test_number = (int) (tp - test_info);

	result = split_qnameval(buf, &key, &value);
	errs += compare(test_number, "result", tp->expect_result, result);
	errs += compare(test_number, "key", tp->expect_key, key);
	errs += compare(test_number, "value", tp->expect_value, value);
	myfree(buf);
    }
    exit(errs);
}

#endif
