/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#ifndef lib_pcap_usb_h
#define lib_pcap_usb_h

#include <pcap/pcap-inttypes.h>

/*
 * possible transfer mode
 */
#define URB_TRANSFER_IN   0x80
#define URB_ISOCHRONOUS   0x0
#define URB_INTERRUPT     0x1
#define URB_CONTROL       0x2
#define URB_BULK          0x3

/*
 * possible event type
 */
#define URB_SUBMIT        'S'
#define URB_COMPLETE      'C'
#define URB_ERROR         'E'

/*
 * USB setup header as defined in USB specification.
 * Appears at the front of each Control S-type packet in DLT_USB captures.
 */
typedef struct _usb_setup {
	uint8_t bmRequestType;
	uint8_t bRequest;
	uint16_t wValue;
	uint16_t wIndex;
	uint16_t wLength;
} pcap_usb_setup;

/*
 * Information from the URB for Isochronous transfers.
 */
typedef struct _iso_rec {
	int32_t	error_count;
	int32_t	numdesc;
} iso_rec;

/*
 * Header prepended by linux kernel to each event.
 * Appears at the front of each packet in DLT_USB_LINUX captures.
 */
typedef struct _usb_header {
	uint64_t id;
	uint8_t event_type;
	uint8_t transfer_type;
	uint8_t endpoint_number;
	uint8_t device_address;
	uint16_t bus_id;
	char setup_flag;/*if !=0 the urb setup header is not present*/
	char data_flag; /*if !=0 no urb data is present*/
	int64_t ts_sec;
	int32_t ts_usec;
	int32_t status;
	uint32_t urb_len;
	uint32_t data_len; /* amount of urb data really present in this event*/
	pcap_usb_setup setup;
} pcap_usb_header;

/*
 * Header prepended by linux kernel to each event for the 2.6.31
 * and later kernels; for the 2.6.21 through 2.6.30 kernels, the
 * "iso_rec" information, and the fields starting with "interval"
 * are zeroed-out padding fields.
 *
 * Appears at the front of each packet in DLT_USB_LINUX_MMAPPED captures.
 */
typedef struct _usb_header_mmapped {
	uint64_t id;
	uint8_t event_type;
	uint8_t transfer_type;
	uint8_t endpoint_number;
	uint8_t device_address;
	uint16_t bus_id;
	char setup_flag;/*if !=0 the urb setup header is not present*/
	char data_flag; /*if !=0 no urb data is present*/
	int64_t ts_sec;
	int32_t ts_usec;
	int32_t status;
	uint32_t urb_len;
	uint32_t data_len; /* amount of urb data really present in this event*/
	union {
		pcap_usb_setup setup;
		iso_rec iso;
	} s;
	int32_t	interval;	/* for Interrupt and Isochronous events */
	int32_t start_frame;	/* for Isochronous events */
	uint32_t xfer_flags;	/* copy of URB's transfer flags */
	uint32_t ndesc;	/* number of isochronous descriptors */
} pcap_usb_header_mmapped;

/*
 * Isochronous descriptors; for isochronous transfers there might be
 * one or more of these at the beginning of the packet data.  The
 * number of descriptors is given by the "ndesc" field in the header;
 * as indicated, in older kernels that don't put the descriptors at
 * the beginning of the packet, that field is zeroed out, so that field
 * can be trusted even in captures from older kernels.
 */
typedef struct _usb_isodesc {
	int32_t		status;
	uint32_t	offset;
	uint32_t	len;
	uint8_t	pad[4];
} usb_isodesc;

#endif
