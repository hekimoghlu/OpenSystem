/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 8, 2024.
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
/* $Id: t_sockaddr.c,v 1.14 2007/06/19 23:47:00 tbox Exp $ */

#include <config.h>

#include <isc/netaddr.h>
#include <isc/result.h>
#include <isc/sockaddr.h>

#include <tests/t_api.h>

static int
test_isc_sockaddr_eqaddrprefix(void) {
	struct in_addr ina_a;
	struct in_addr ina_b;
	struct in_addr ina_c;
	isc_sockaddr_t isa_a;
	isc_sockaddr_t isa_b;
	isc_sockaddr_t isa_c;

	if (inet_pton(AF_INET, "194.100.32.87", &ina_a) < 0)
		return T_FAIL;
	if (inet_pton(AF_INET, "194.100.32.80", &ina_b) < 0)
		return T_FAIL;
	if (inet_pton(AF_INET, "194.101.32.87", &ina_c) < 0)
		return T_FAIL;
	isc_sockaddr_fromin(&isa_a, &ina_a, 0);
	isc_sockaddr_fromin(&isa_b, &ina_b, 42);
	isc_sockaddr_fromin(&isa_c, &ina_c, 0);

	if (isc_sockaddr_eqaddrprefix(&isa_a, &isa_b, 0) != ISC_TRUE)
		return T_FAIL;
	if (isc_sockaddr_eqaddrprefix(&isa_a, &isa_b, 29) != ISC_TRUE)
		return T_FAIL;
	if (isc_sockaddr_eqaddrprefix(&isa_a, &isa_b, 30) != ISC_FALSE)
		return T_FAIL;
	if (isc_sockaddr_eqaddrprefix(&isa_a, &isa_b, 32) != ISC_FALSE)
		return T_FAIL;
	if (isc_sockaddr_eqaddrprefix(&isa_a, &isa_c, 8) != ISC_TRUE)
		return T_FAIL;
	if (isc_sockaddr_eqaddrprefix(&isa_a, &isa_c, 16) != ISC_FALSE)
		return T_FAIL;

	return T_PASS;
}

static void
t1(void) {
	int result;
	t_assert("isc_sockaddr_eqaddrprefix", 1, T_REQUIRED,
		 "isc_sockaddr_eqaddrprefix() returns ISC_TRUE when "
		 "prefixes of a and b are equal, and ISC_FALSE when "
		 "they are not equal");
	result = test_isc_sockaddr_eqaddrprefix();
	t_result(result);
}

static int
test_isc_netaddr_masktoprefixlen(void) {
	struct in_addr na_a;
	struct in_addr na_b;
	struct in_addr na_c;
	struct in_addr na_d;
	isc_netaddr_t ina_a;
	isc_netaddr_t ina_b;
	isc_netaddr_t ina_c;
	isc_netaddr_t ina_d;
	unsigned int plen;

	if (inet_pton(AF_INET, "0.0.0.0", &na_a) < 0)
		return T_FAIL;
	if (inet_pton(AF_INET, "255.255.255.254", &na_b) < 0)
		return T_FAIL;
	if (inet_pton(AF_INET, "255.255.255.255", &na_c) < 0)
		return T_FAIL;
	if (inet_pton(AF_INET, "255.255.255.0", &na_d) < 0)
		return T_FAIL;
	isc_netaddr_fromin(&ina_a, &na_a);
	isc_netaddr_fromin(&ina_b, &na_b);
	isc_netaddr_fromin(&ina_c, &na_c);
	isc_netaddr_fromin(&ina_d, &na_d);

	if (isc_netaddr_masktoprefixlen(&ina_a, &plen) != ISC_R_SUCCESS)
		return T_FAIL;
	if (plen != 0)
		return T_FAIL;

	if (isc_netaddr_masktoprefixlen(&ina_b, &plen) != ISC_R_SUCCESS)
		return T_FAIL;
	if (plen != 31)
		return T_FAIL;

	if (isc_netaddr_masktoprefixlen(&ina_c, &plen) != ISC_R_SUCCESS)
		return T_FAIL;
	if (plen != 32)
		return T_FAIL;

	if (isc_netaddr_masktoprefixlen(&ina_d, &plen) != ISC_R_SUCCESS)
		return T_FAIL;
	if (plen != 24)
		return T_FAIL;

	return T_PASS;
}

static void
t2(void) {
	int result;
	t_assert("isc_netaddr_masktoprefixlen", 1, T_REQUIRED,
		 "isc_netaddr_masktoprefixlen() calculates "
		 "correct prefix lengths ");
	result = test_isc_netaddr_masktoprefixlen();
	t_result(result);
}

testspec_t	T_testlist[] = {
	{	(PFV) t1,	"isc_sockaddr_eqaddrprefix"	},
	{	(PFV) t2,	"isc_netaddr_masktoprefixlen"	},
	{	(PFV) 0,	NULL				}
};

#ifdef WIN32
int
main(int argc, char **argv) {
	t_settests(T_testlist);
	return (t_main(argc, argv));
}
#endif
