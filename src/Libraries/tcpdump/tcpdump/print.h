/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
#ifndef print_h
#define print_h

void	init_print(netdissect_options *ndo, uint32_t localnet, uint32_t mask);

int	has_printer(int type);

if_printer get_if_printer(int type);

void	pretty_print_packet(netdissect_options *ndo,
	    const struct pcap_pkthdr *h, const u_char *sp,
	    u_int packets_captured);

void	ndo_set_function_pointers(netdissect_options *ndo);

#ifdef __APPLE__
void print_pcap(u_char *, const struct pcap_pkthdr *, const u_char *);
void print_pcap_ng_block(u_char *, const struct pcap_pkthdr *, const u_char *);
void print_pktap_packet(u_char *, const struct pcap_pkthdr *, const u_char *);
#endif /* __APPLE__ */

#endif /* print_h */
