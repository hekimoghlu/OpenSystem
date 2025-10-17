/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/udp.h>
#include <netinet/in_systm.h>
#include <netinet/ip.h>
#include <netinet/bootp.h>
#include <arpa/inet.h>
#include <net/if_arp.h>
#include <string.h>
#include "dhcplib.h"
#include "cfutil.h"
#include "symbol_scope.h"

PRIVATE_EXTERN void
dhcp_packet_print_cfstr(CFMutableStringRef str, struct dhcp * dp, int pkt_len)
{
    dhcpol_t 		options;

    if (pkt_len < sizeof(struct dhcp)) {
	STRING_APPEND(str, "Packet is too short %d < %d\n", pkt_len,
		(int)sizeof(struct dhcp));
	return;
    }
    dhcpol_init(&options);
    dhcpol_parse_packet(&options, dp, pkt_len, NULL);
    dhcp_packet_with_options_print_cfstr(str, dp, pkt_len, &options);
    dhcpol_free(&options);
    return;
}

PRIVATE_EXTERN void
dhcp_packet_with_options_print_cfstr(CFMutableStringRef str,
				     struct dhcp * dp, int pkt_len,
				     dhcpol_t * options)
{
    int			hlen;
    boolean_t		invalid_hlen = FALSE;
    char		ntopbuf[INET_ADDRSTRLEN];

    STRING_APPEND(str, "op = ");
    if (dp->dp_op == BOOTREQUEST) {
	STRING_APPEND(str, "BOOTREQUEST\n");
    }
    else if (dp->dp_op == BOOTREPLY) {
	STRING_APPEND(str, "BOOTREPLY\n");
    }
    else {
	STRING_APPEND(str, "OP(%d)\n", (int)dp->dp_op);
    }
    STRING_APPEND(str, "htype = %d\n", (int)dp->dp_htype);
    STRING_APPEND(str, "flags = 0x%x\n", ntohs(dp->dp_flags));
    hlen = dp->dp_hlen;
    if (hlen > sizeof(dp->dp_chaddr)) {
	STRING_APPEND(str, "hlen = %d (invalid > %lu)\n",
		      hlen, sizeof(dp->dp_chaddr));
	hlen = sizeof(dp->dp_chaddr);
	invalid_hlen = TRUE;
    }
    else {
	STRING_APPEND(str, "hlen = %d\n", hlen);
    }
    STRING_APPEND(str, "hops = %d\n", (int)dp->dp_hops);
    STRING_APPEND(str, "xid = 0x%lx\n", (u_long)ntohl(dp->dp_xid));
    STRING_APPEND(str, "secs = %hu\n", ntohs(dp->dp_secs));
    STRING_APPEND(str, "ciaddr = %s\n",
		  inet_ntop(AF_INET, &dp->dp_ciaddr, ntopbuf, sizeof(ntopbuf)));
    STRING_APPEND(str, "yiaddr = %s\n",
		  inet_ntop(AF_INET, &dp->dp_yiaddr, ntopbuf, sizeof(ntopbuf)));
    STRING_APPEND(str, "siaddr = %s\n",
		  inet_ntop(AF_INET, &dp->dp_siaddr, ntopbuf, sizeof(ntopbuf)));
    STRING_APPEND(str, "giaddr = %s\n",
		  inet_ntop(AF_INET, &dp->dp_giaddr, ntopbuf, sizeof(ntopbuf)));
    STRING_APPEND(str, "chaddr = %s", invalid_hlen ? "[truncated] " : "");
    for (int i = 0; i < hlen; i++) {
	if (i != 0) {
	    STRING_APPEND(str, ":");
	}
	STRING_APPEND(str, "%0x", (int)dp->dp_chaddr[i]);
    }
    STRING_APPEND(str, "\n");
    STRING_APPEND(str, "sname = %.*s\n", (int)sizeof(dp->dp_sname),
		  dp->dp_sname);
    STRING_APPEND(str, "file = %.*s\n", (int)sizeof(dp->dp_file),
		  dp->dp_file);
    if (options != NULL && dhcpol_count(options) > 0) {
	STRING_APPEND(str, "options:\n");
	dhcpol_print_cfstr(str, options);
    }
    return;
}

PRIVATE_EXTERN void
dhcp_packet_fprint(FILE * f, struct dhcp * dp, int pkt_len)
{
    CFMutableStringRef	str;

    str = CFStringCreateMutable(NULL, 0);
    dhcp_packet_print_cfstr(str, dp, pkt_len);
    my_CFStringPrint(f, str);
    CFRelease(str);
    fflush(f);
    return;
}

PRIVATE_EXTERN void
dhcp_packet_print(struct dhcp *dp, int pkt_len)
{
    dhcp_packet_fprint(stdout, dp, pkt_len);
    return;
}

PRIVATE_EXTERN boolean_t
dhcp_packet_match(struct bootp * packet, u_int32_t xid, 
		  u_char hwtype, void * hwaddr, int hwlen)
{
    int		check_len;

    switch (hwtype) {
    default:
    case ARPHRD_ETHER:
	check_len = hwlen;
	break;
    case ARPHRD_IEEE1394:
	check_len = 0;
	break;
    }
    if (packet->bp_op != BOOTREPLY
	|| ntohl(packet->bp_xid) != xid
	|| (packet->bp_htype != hwtype)
	|| (packet->bp_hlen != check_len)
	|| (check_len != 0 && bcmp(packet->bp_chaddr, hwaddr, check_len))) {
	return (FALSE);
    }
    return (TRUE);
}
