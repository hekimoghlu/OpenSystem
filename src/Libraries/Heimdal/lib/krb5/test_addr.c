/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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

static void
print_addr(krb5_context context, const char *addr)
{
    krb5_addresses addresses;
    krb5_error_code ret;
    char buf[38];
    char buf2[1000];
    size_t len;
    int i;

    ret = krb5_parse_address(context, addr, &addresses);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_address");

    if (addresses.len < 1)
	krb5_err(context, 1, ret, "too few addresses");

    for (i = 0; i < addresses.len; i++) {
	krb5_print_address(&addresses.val[i], buf, sizeof(buf), &len);
#if 0
	printf("addr %d: %s (%d/%d)\n", i, buf, (int)len, (int)strlen(buf));
#endif
	if (strlen(buf) > sizeof(buf))
	    krb5_err(context, 1, ret, "len %d larger then buf %d",
		     (int)strlen(buf), (int)sizeof(buf));
	krb5_print_address(&addresses.val[i], buf2, sizeof(buf2), &len);
#if 0
	printf("addr %d: %s (%d/%d)\n", i, buf2, (int)len, (int)strlen(buf2));
#endif
	if (strlen(buf2) > sizeof(buf2))
	    krb5_err(context, 1, ret, "len %d larger then buf %d",
		     (int)strlen(buf2), (int)sizeof(buf2));

    }
    krb5_free_addresses(context, &addresses);

}

static void
truncated_addr(krb5_context context, const char *addr,
	       size_t truncate_len, size_t outlen)
{
    krb5_addresses addresses;
    krb5_error_code ret;
    char *buf;
    size_t len;

    buf = ecalloc(1, outlen + 1);

    ret = krb5_parse_address(context, addr, &addresses);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_address");

    if (addresses.len != 1)
	krb5_err(context, 1, ret, "addresses should be one");

    krb5_print_address(&addresses.val[0], buf, truncate_len, &len);

#if 0
    printf("addr %s (%d/%d) should be %d\n", buf, (int)len, (int)strlen(buf), (int)outlen);
#endif

    if (truncate_len > strlen(buf) + 1)
	krb5_err(context, 1, ret, "%s truncate_len %d larger then strlen %d source %s",
		 buf, (int)truncate_len, (int)strlen(buf), addr);

    if (outlen != len)
	krb5_err(context, 1, ret, "%s: outlen %d != len %d",
		 buf, (int)outlen, (int)strlen(buf));

    krb5_print_address(&addresses.val[0], buf, outlen + 1, &len);

#if 0
    printf("addr %s (%d/%d)\n", buf, (int)len, (int)strlen(buf));
#endif

    if (len != outlen)
	abort();
    if (strlen(buf) != len)
	abort();

    krb5_free_addresses(context, &addresses);
    free(buf);
}

static void
check_truncation(krb5_context context, const char *addr)
{
    int i, len = strlen(addr);

    truncated_addr(context, addr, len, len);

    for (i = 0; i < len; i++)
	truncated_addr(context, addr, i, len);
}

static void
match_addr(krb5_context context, const char *range_addr,
	   const char *one_addr, int match)
{
    krb5_addresses range, one;
    krb5_error_code ret;

    ret = krb5_parse_address(context, range_addr, &range);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_address");

    if (range.len != 1)
	krb5_err(context, 1, ret, "wrong num of addresses");

    ret = krb5_parse_address(context, one_addr, &one);
    if (ret)
	krb5_err(context, 1, ret, "krb5_parse_address");

    if (one.len != 1)
	krb5_err(context, 1, ret, "wrong num of addresses");

    if (krb5_address_order(context, &range.val[0], &one.val[0]) == 0) {
	if (!match)
	    krb5_errx(context, 1, "match when one shouldn't be");
    } else {
	if (match)
	    krb5_errx(context, 1, "no match when one should be");
    }

    krb5_free_addresses(context, &range);
    krb5_free_addresses(context, &one);
}

#ifdef _MSC_VER

/* For the truncation tests, calling strcpy_s() or strcat_s() with a
   size of 0 results in the invalid parameter handler being invoked.
   For the debug version, the runtime also throws an assert. */

static void
inv_param_handler(const wchar_t* expression,
		  const wchar_t* function,
		  const wchar_t* file,
		  unsigned int line,
		  uintptr_t pReserved)
{
    printf("Invalid parameter handler invoked for: %S in %S(%d) [%S]\n",
	   function, file, line, expression);
}

static _invalid_parameter_handler _inv_old = NULL;

#define SET_INVALID_PARAM_HANDLER _inv_old = _set_invalid_parameter_handler(inv_param_handler)

#else

#define SET_INVALID_PARAM_HANDLER ((void) 0)

#endif

int
main(int argc, char **argv)
{
    krb5_context context;
    krb5_error_code ret;

    SET_INVALID_PARAM_HANDLER;

    setprogname(argv[0]);

    ret = krb5_init_context(&context);
    if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    print_addr(context, "RANGE:127.0.0.0/8");
    print_addr(context, "RANGE:127.0.0.0/24");
    print_addr(context, "RANGE:IPv4:127.0.0.0-IPv4:127.0.0.255");
    print_addr(context, "RANGE:130.237.237.4/29");
#ifdef HAVE_IPV6
    print_addr(context, "RANGE:2001:db8:1:2:3:4:1428:7ab/64");
    print_addr(context, "RANGE:IPv6:fe80::209:6bff:fea0:e522/64");
    print_addr(context, "RANGE:IPv6:fe80::-IPv6:fe80::ffff:ffff:ffff:ffff");
    print_addr(context, "RANGE:fe80::-fe80::ffff:ffff:ffff:ffff");
#endif

    check_truncation(context, "IPv4:127.0.0.0");
    check_truncation(context, "RANGE:IPv4:127.0.0.0-IPv4:127.0.0.255");
#ifdef HAVE_IPV6
    check_truncation(context, "IPv6:::");
    check_truncation(context, "IPv6:::1");
    check_truncation(context, "IPv6:2001:db8:1:2:3:4:1428:7ab");
    check_truncation(context, "IPv6:fe80::209:0:0:0");
    check_truncation(context, "IPv6:fe80::ffff:ffff:ffff:ffff");
#endif

    match_addr(context, "RANGE:127.0.0.0/8", "inet:127.0.0.0", 1);
    match_addr(context, "RANGE:127.0.0.0/8", "inet:127.255.255.255", 1);
    match_addr(context, "RANGE:127.0.0.0/8", "inet:128.0.0.0", 0);

    match_addr(context, "RANGE:130.237.237.8/29", "inet:130.237.237.7", 0);
    match_addr(context, "RANGE:130.237.237.8/29", "inet:130.237.237.8", 1);
    match_addr(context, "RANGE:130.237.237.8/29", "inet:130.237.237.15", 1);
    match_addr(context, "RANGE:130.237.237.8/29", "inet:130.237.237.16", 0);

    krb5_free_context(context);

    return 0;
}
