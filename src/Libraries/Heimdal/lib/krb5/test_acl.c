/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
#include "krb5_locl.h"
#include <err.h>

#define RETVAL(c, r, e, s) \
	do { if (r != e) krb5_errx(c, 1, "%s", s); } while (0)
#define STRINGMATCH(c, s, _s1, _s2) \
	do {							\
		if (_s1 == NULL || _s2 == NULL) 		\
			krb5_errx(c, 1, "s1 or s2 is NULL");	\
		if (strcmp(_s1,_s2) != 0) 			\
			krb5_errx(c, 1, "%s", s);		\
	} while (0)

static void
test_match_string(krb5_context context)
{
    krb5_error_code ret;
    char *s1, *s2;

    ret = krb5_acl_match_string(context, "foo", "s", "foo");
    RETVAL(context, ret, 0, "single s");
    ret = krb5_acl_match_string(context, "foo foo", "s", "foo");
    RETVAL(context, ret, EACCES, "too many strings");
    ret = krb5_acl_match_string(context, "foo bar", "ss", "foo", "bar");
    RETVAL(context, ret, 0, "two strings");
    ret = krb5_acl_match_string(context, "foo  bar", "ss", "foo", "bar");
    RETVAL(context, ret, 0, "two strings double space");
    ret = krb5_acl_match_string(context, "foo \tbar", "ss", "foo", "bar");
    RETVAL(context, ret, 0, "two strings space + tab");
    ret = krb5_acl_match_string(context, "foo", "ss", "foo", "bar");
    RETVAL(context, ret, EACCES, "one string, two format strings");
    ret = krb5_acl_match_string(context, "foo", "ss", "foo", "foo");
    RETVAL(context, ret, EACCES, "one string, two format strings (same)");
    ret = krb5_acl_match_string(context, "foo  \t", "s", "foo");
    RETVAL(context, ret, 0, "ending space");

    ret = krb5_acl_match_string(context, "foo/bar", "f", "foo/bar");
    RETVAL(context, ret, 0, "liternal fnmatch");
    ret = krb5_acl_match_string(context, "foo/bar", "f", "foo/*");
    RETVAL(context, ret, 0, "foo/*");
    ret = krb5_acl_match_string(context, "foo/bar.example.org", "f",
				"foo/*.example.org");
    RETVAL(context, ret, 0, "foo/*.example.org");
    ret = krb5_acl_match_string(context, "foo/bar.example.com", "f",
				"foo/*.example.org");
    RETVAL(context, ret, EACCES, "foo/*.example.com");

    ret = krb5_acl_match_string(context, "foo/bar/baz", "f", "foo/*/baz");
    RETVAL(context, ret, 0, "foo/*/baz");

    ret = krb5_acl_match_string(context, "foo", "r", &s1);
    RETVAL(context, ret, 0, "ret 1");
    STRINGMATCH(context, "ret 1 match", s1, "foo"); free(s1);

    ret = krb5_acl_match_string(context, "foo bar", "rr", &s1, &s2);
    RETVAL(context, ret, 0, "ret 2");
    STRINGMATCH(context, "ret 2 match 1", s1, "foo"); free(s1);
    STRINGMATCH(context, "ret 2 match 2", s2, "bar"); free(s2);

    ret = krb5_acl_match_string(context, "foo bar", "sr", "bar", &s1);
    RETVAL(context, ret, EACCES, "ret mismatch");
    if (s1 != NULL) krb5_errx(context, 1, "s1 not NULL");

    ret = krb5_acl_match_string(context, "foo", "l", "foo");
    RETVAL(context, ret, EINVAL, "unknown letter");
}


int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;

    setprogname(argv[0]);

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    test_match_string(context);

    krb5_free_context(context);

    return 0;
}
