/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#ifndef lib_pcap_namedb_h
#define lib_pcap_namedb_h

#ifdef __APPLE__
#include <stdio.h>		/* FILE */
#endif /* __APPLE__ */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * As returned by the pcap_next_etherent()
 * XXX this stuff doesn't belong in this interface, but this
 * library already must do name to address translation, so
 * on systems that don't have support for /etc/ethers, we
 * export these hooks since they're already being used by
 * some applications (such as tcpdump) and already being
 * marked as exported in some OSes offering libpcap (such
 * as Debian).
 */
struct pcap_etherent {
	u_char addr[6];
	char name[122];
};
#ifndef PCAP_ETHERS_FILE
#define PCAP_ETHERS_FILE "/etc/ethers"
#endif
PCAP_AVAILABLE_0_4
PCAP_API struct	pcap_etherent *pcap_next_etherent(FILE *);

PCAP_AVAILABLE_0_4
PCAP_API u_char *pcap_ether_hostton(const char*);

PCAP_AVAILABLE_0_4
PCAP_API u_char *pcap_ether_aton(const char *);

PCAP_AVAILABLE_0_4
PCAP_API bpf_u_int32 **pcap_nametoaddr(const char *)
PCAP_DEPRECATED(pcap_nametoaddr, "this is not reentrant; use 'pcap_nametoaddrinfo' instead");

PCAP_AVAILABLE_0_4
PCAP_API struct addrinfo *pcap_nametoaddrinfo(const char *);

PCAP_AVAILABLE_0_4
PCAP_API bpf_u_int32 pcap_nametonetaddr(const char *);

PCAP_AVAILABLE_0_4
PCAP_API int	pcap_nametoport(const char *, int *, int *);

PCAP_AVAILABLE_0_4
PCAP_API int	pcap_nametoportrange(const char *, int *, int *, int *);

PCAP_AVAILABLE_0_4
PCAP_API int	pcap_nametoproto(const char *);

PCAP_AVAILABLE_0_4
PCAP_API int	pcap_nametoeproto(const char *);

PCAP_AVAILABLE_0_4
PCAP_API int	pcap_nametollc(const char *);
/*
 * If a protocol is unknown, PROTO_UNDEF is returned.
 * Also, pcap_nametoport() returns the protocol along with the port number.
 * If there are ambiguous entried in /etc/services (i.e. domain
 * can be either tcp or udp) PROTO_UNDEF is returned.
 */
#define PROTO_UNDEF		-1

#ifdef __cplusplus
}
#endif

#endif
