/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#ifndef _S_DHCPLIB_H
#define _S_DHCPLIB_H

#include <mach/boolean.h>
#include <sys/time.h>
#import <netinet/bootp.h>
#include "dhcp_options.h"
#include "gen_dhcp_tags.h"

void	dhcp_packet_print_cfstr(CFMutableStringRef str, 
				struct dhcp * dp, int pkt_len);
void	dhcp_packet_with_options_print_cfstr(CFMutableStringRef str,
					     struct dhcp * dp, int pkt_len,
					     dhcpol_t * options);
void	dhcp_packet_fprint(FILE * f, struct dhcp * dp, int pkt_len);
void	dhcp_packet_print(struct dhcp * dp, int pkt_len);

/*
 * Function: is_dhcp_packet
 *
 * Purpose:
 *   Return whether packet is a DHCP packet.
 *   If the packet contains DHCP message ids, then its a DHCP packet.
 */
static __inline__ boolean_t
is_dhcp_packet(dhcpol_t * options, dhcp_msgtype_t * msgtype)
{
    if (options) {
	u_char * opt;
	int opt_len;

	opt = dhcpol_find(options, dhcptag_dhcp_message_type_e,
			  &opt_len, NULL);
	if (opt != NULL) {
	    if (msgtype)
		*msgtype = *opt;
	    return (TRUE);
	}
    }
    return (FALSE);
}

boolean_t
dhcp_packet_match(struct bootp * packet, u_int32_t xid, 
		  u_char hwtype, void * hwaddr, int hwlen);

#endif /* _S_DHCPLIB_H */
