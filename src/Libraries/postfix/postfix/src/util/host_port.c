/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#include <ctype.h>

/* Utility library. */

#include <msg.h>
#include <split_at.h>
#include <stringops.h>			/* XXX util_utf8_enable */
#include <valid_utf8_hostname.h>

/* Global library. */

#include <host_port.h>

 /*
  * Point-fix workaround. The libutil library should be email agnostic, but
  * we can't rip up the library APIs in the stable releases.
  */
#include <string.h>
#ifdef STRCASECMP_IN_STRINGS_H
#include <strings.h>
#endif
#define IPV6_COL           "IPv6:"	/* RFC 2821 */
#define IPV6_COL_LEN       (sizeof(IPV6_COL) - 1)
#define HAS_IPV6_COL(str)  (strncasecmp((str), IPV6_COL, IPV6_COL_LEN) == 0)

/* host_port - parse string into host and port, destroy string */

const char *host_port(char *buf, char **host, char *def_host,
		              char **port, char *def_service)
{
    char   *cp = buf;
    int     ipv6 = 0;

    /*-
     * [host]:port, [host]:, [host].
     * [ipv6:ipv6addr]:port, [ipv6:ipv6addr]:, [ipv6:ipv6addr].
     */
    if (*cp == '[') {
	++cp;
	if ((ipv6 = HAS_IPV6_COL(cp)) != 0)
	    cp += IPV6_COL_LEN;
	*host = cp;
	if ((cp = split_at(cp, ']')) == 0)
	    return ("missing \"]\"");
	if (*cp && *cp++ != ':')
	    return ("garbage after \"]\"");
	if (ipv6 && !valid_ipv6_hostaddr(*host, DONT_GRIPE))
	    return ("malformed IPv6 address");
	*port = *cp ? cp : def_service;
    }

    /*
     * host:port, host:, host, :port, port.
     */
    else {
	if ((cp = split_at_right(buf, ':')) != 0) {
	    *host = *buf ? buf : def_host;
	    *port = *cp ? cp : def_service;
	} else {
	    *host = def_host ? def_host : (*buf ? buf : 0);
	    *port = def_service ? def_service : (*buf ? buf : 0);
	}
    }
    if (*host == 0)
	return ("missing host information");
    if (*port == 0)
	return ("missing service information");

    /*
     * Final sanity checks. We're still sloppy, allowing bare numerical
     * network addresses instead of requiring proper [ipaddress] forms.
     */
    if (*host != def_host 
	&& !valid_utf8_hostname(util_utf8_enable, *host, DONT_GRIPE)
	&& !valid_hostaddr(*host, DONT_GRIPE))
	return ("valid hostname or network address required");
    if (*port != def_service && ISDIGIT(**port) && !alldig(*port))
	return ("garbage after numerical service");
    return (0);
}

#ifdef TEST

#include <vstream.h>
#include <vstring.h>
#include <vstring_vstream.h>

#define STR(x) vstring_str(x)

int     main(int unused_argc, char **unused_argv)
{
    VSTRING *in_buf = vstring_alloc(10);
    VSTRING *parse_buf = vstring_alloc(10);
    char   *host;
    char   *port;
    const char *err;

    while (vstring_fgets_nonl(in_buf, VSTREAM_IN)) {
	vstream_printf(">> %s\n", STR(in_buf));
	vstream_fflush(VSTREAM_OUT);
	if (*STR(in_buf) == '#')
	    continue;
	vstring_strcpy(parse_buf, STR(in_buf));
	if ((err = host_port(STR(parse_buf), &host, (char *) 0, &port, "default-service")) != 0) {
	    msg_warn("%s in %s", err, STR(in_buf));
	} else {
	    vstream_printf("host %s port %s\n", host, port);
	    vstream_fflush(VSTREAM_OUT);
	}
    }
    vstring_free(in_buf);
    vstring_free(parse_buf);
    return (0);
}

#endif
