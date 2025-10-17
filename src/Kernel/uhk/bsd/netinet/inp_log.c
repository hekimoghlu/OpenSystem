/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#include <netinet/in_var.h>
#include <netinet/inp_log.h>

SYSCTL_NODE(_net_inet_ip, OID_AUTO, log, CTLFLAG_RW | CTLFLAG_LOCKED, 0,
    "TCP/IP + UDP logs");

#if (DEVELOPMENT || DEBUG)
#define INP_LOG_PRIVACY_DEFAULT 0
#else
#define INP_LOG_PRIVACY_DEFAULT 1
#endif /* (DEVELOPMENT || DEBUG) */

int inp_log_privacy = INP_LOG_PRIVACY_DEFAULT;
SYSCTL_INT(_net_inet_ip_log, OID_AUTO, privacy,
    CTLFLAG_RW | CTLFLAG_LOCKED, &inp_log_privacy, 0, "");

void
inp_log_addresses(struct inpcb *inp, char *__sized_by(lbuflen) lbuf,
    socklen_t lbuflen, char *__sized_by(fbuflen) fbuf,
    socklen_t fbuflen)
{
	/*
	 * Ugly but %{private} does not work in the kernel version of os_log()
	 */
	if (inp_log_privacy != 0) {
		if (inp->inp_vflag & INP_IPV6) {
			strlcpy(lbuf, "<IPv6-redacted>", lbuflen);
			strlcpy(fbuf, "<IPv6-redacted>", fbuflen);
		} else {
			strlcpy(lbuf, "<IPv4-redacted>", lbuflen);
			strlcpy(fbuf, "<IPv4-redacted>", fbuflen);
		}
	} else if (inp->inp_vflag & INP_IPV6) {
		struct in6_addr addr6;

		if (IN6_IS_ADDR_LINKLOCAL(&inp->in6p_laddr)) {
			addr6 = inp->in6p_laddr;
			addr6.s6_addr16[1] = 0;
			inet_ntop(AF_INET6, (void *)&addr6, lbuf, lbuflen);
		} else {
			inet_ntop(AF_INET6, (void *)&inp->in6p_laddr, lbuf, lbuflen);
		}

		if (IN6_IS_ADDR_LINKLOCAL(&inp->in6p_faddr)) {
			addr6 = inp->in6p_faddr;
			addr6.s6_addr16[1] = 0;
			inet_ntop(AF_INET6, (void *)&addr6, fbuf, fbuflen);
		} else {
			inet_ntop(AF_INET6, (void *)&inp->in6p_faddr, fbuf, fbuflen);
		}
	} else {
		inet_ntop(AF_INET, (void *)&inp->inp_laddr.s_addr, lbuf, lbuflen);
		inet_ntop(AF_INET, (void *)&inp->inp_faddr.s_addr, fbuf, fbuflen);
	}
}
