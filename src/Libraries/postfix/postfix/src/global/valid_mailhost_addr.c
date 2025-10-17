/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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
/* System library. */

#include <sys_defs.h>
#include <string.h>

#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif

/* Utility library. */

#include <msg.h>
#include <myaddrinfo.h>

/* Global library. */

#include <valid_mailhost_addr.h>

/* Application-specific. */

#define IPV6_COL_LEN       (sizeof(IPV6_COL) - 1)
#define HAS_IPV6_COL(str)  (strncasecmp((str), IPV6_COL, IPV6_COL_LEN) == 0)
#define SKIP_IPV6_COL(str) (HAS_IPV6_COL(str) ? (str) + IPV6_COL_LEN : (str))

/* valid_mailhost_addr - validate RFC 2821 numerical address form */

const char *valid_mailhost_addr(const char *addr, int gripe)
{
    const char *bare_addr;

    bare_addr = SKIP_IPV6_COL(addr);
    return ((bare_addr != addr ? valid_ipv6_hostaddr : valid_ipv4_hostaddr)
	    (bare_addr, gripe) ? bare_addr : 0);
}

/* valid_mailhost_literal - validate [RFC 2821 numerical address] form */

int     valid_mailhost_literal(const char *addr, int gripe)
{
    const char *myname = "valid_mailhost_literal";
    MAI_HOSTADDR_STR hostaddr;
    const char *last;
    size_t address_bytes;

    if (*addr != '[') {
	if (gripe)
	    msg_warn("%s: '[' expected at start: %.100s", myname, addr);
	return (0);
    }
    if ((last = strchr(addr, ']')) == 0) {
	if (gripe)
	    msg_warn("%s: ']' expected at end: %.100s", myname, addr);
	return (0);
    }
    if (last[1]) {
	if (gripe)
	    msg_warn("%s: unexpected text after ']': %.100s", myname, addr);
	return (0);
    }
    if ((address_bytes = last - addr - 1) >= sizeof(hostaddr.buf)) {
	if (gripe)
	    msg_warn("%s: too much text: %.100s", myname, addr);
	return (0);
    }
    strncpy(hostaddr.buf, addr + 1, address_bytes);
    hostaddr.buf[address_bytes] = 0;
    return (valid_mailhost_addr(hostaddr.buf, gripe) != 0);
}

#ifdef TEST

 /*
  * Test program - reads hostnames from stdin, reports invalid hostnames to
  * stderr.
  */
#include <stdlib.h>

#include <vstring.h>
#include <vstream.h>
#include <vstring_vstream.h>
#include <msg_vstream.h>

int     main(int unused_argc, char **argv)
{
    VSTRING *buffer = vstring_alloc(1);

    msg_vstream_init(argv[0], VSTREAM_ERR);
    msg_verbose = 1;

    while (vstring_fgets_nonl(buffer, VSTREAM_IN)) {
	msg_info("testing: \"%s\"", vstring_str(buffer));
	if (vstring_str(buffer)[0] == '[')
	    valid_mailhost_literal(vstring_str(buffer), DO_GRIPE);
	else
	    valid_mailhost_addr(vstring_str(buffer), DO_GRIPE);
    }
    exit(0);
}

#endif
