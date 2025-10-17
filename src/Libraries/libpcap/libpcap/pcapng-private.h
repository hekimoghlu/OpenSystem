/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#ifndef libpcapng_pcapng_private_h
#define libpcapng_pcapng_private_h

#include <sys/queue.h>

#include "pcap/pcap-ng.h"

struct pcapng_block {
	u_char		*pcapng_bufptr;
	size_t		pcapng_buflen;
	int		pcapng_buf_is_external;

	uint32_t	pcapng_block_type;
	size_t		pcapng_block_len;
	int		pcapng_block_swapped;

	size_t		pcapng_fields_len;

	u_char		*pcapng_data_ptr;
	size_t		pcapng_data_len;
	u_int32_t	pcapng_cap_len;
	int		pcapng_data_is_external;

	size_t		pcapng_records_len;
	size_t		pcapng_options_len;

	union {
		struct pcapng_section_header_fields		_section_header;
		struct pcapng_interface_description_fields	_interface_description;
		struct pcapng_packet_fields			_packet;
		struct pcapng_simple_packet_fields		_simple_packet;
		struct pcapng_interface_statistics_fields	_interface_statistics;
		struct pcapng_enhanced_packet_fields		_enhanced_packet;
		struct pcapng_process_information_fields	_process_information;
		struct pcapng_os_event_fields			_os_event_information;
		struct pcapng_decryption_secrets_fields		_decryption_secrets;
	} block_fields_;
};

#define pcap_ng_shb_fields		block_fields_._section_header
#define pcap_ng_idb_fields		block_fields_._interface_description
#define pcap_ng_opb_fields		block_fields_._packet
#define pcap_ng_spb_fields		block_fields_._simple_packet
#define pcap_ng_isb_fields		block_fields_._interface_statistics
#define pcap_ng_epb_fields		block_fields_._enhanced_packet
#define pcap_ng_pib_fields		block_fields_._process_information
#define pcap_ng_osev_fields		block_fields_._os_event_information
#define pcap_ng_dsb_fields		block_fields_._decryption_secrets

/* Representation of on file data structure items */
#define PCAPNG_BYTE_ORDER_MAGIC	0x1A2B3C4D
#define PCAPNG_MAJOR_VERSION	1
#define PCAPNG_MINOR_VERSION	0

#endif
