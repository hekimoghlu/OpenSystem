/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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
#ifndef lib_pcap_bluetooth_h
#define lib_pcap_bluetooth_h

#include <pcap/pcap-inttypes.h>

/*
 * Header prepended libpcap to each bluetooth h4 frame,
 * fields are in network byte order
 */
typedef struct _pcap_bluetooth_h4_header {
	uint32_t direction; /* if first bit is set direction is incoming */
} pcap_bluetooth_h4_header;

/*
 * Header prepended libpcap to each bluetooth linux monitor frame,
 * fields are in network byte order
 */
typedef struct _pcap_bluetooth_linux_monitor_header {
	uint16_t adapter_id;
	uint16_t opcode;
} pcap_bluetooth_linux_monitor_header;

#endif
