/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#ifndef pcap_rpcap_h
#define	pcap_rpcap_h

/*
 * Internal interfaces for "pcap_open()".
 */
pcap_t	*pcap_open_rpcap(const char *source, int snaplen, int flags,
    int read_timeout, struct pcap_rmtauth *auth, char *errbuf);

/*
 * Internal interfaces for "pcap_findalldevs_ex()".
 */
int	pcap_findalldevs_ex_remote(const char *source,
    struct pcap_rmtauth *auth, pcap_if_t **alldevs, char *errbuf);

#endif
