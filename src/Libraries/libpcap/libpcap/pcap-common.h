/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
/*
 * We use the "receiver-makes-right" approach to byte order,
 * because time is at a premium when we are writing the file.
 * In other words, the pcap_file_header and pcap_pkthdr,
 * records are written in host byte order.
 * Note that the bytes of packet data are written out in the order in
 * which they were received, so multi-byte fields in packets are not
 * written in host byte order, they're written in whatever order the
 * sending machine put them in.
 *
 * ntoh[ls] aren't sufficient because we might need to swap on a big-endian
 * machine (if the file was written in little-end order).
 */
#define	SWAPLONG(y) \
    (((((u_int)(y))&0xff)<<24) | \
     ((((u_int)(y))&0xff00)<<8) | \
     ((((u_int)(y))&0xff0000)>>8) | \
     ((((u_int)(y))>>24)&0xff))
#define	SWAPSHORT(y) \
     ((u_short)(((((u_int)(y))&0xff)<<8) | \
                ((((u_int)(y))&0xff00)>>8)))

#ifdef __APPLE__
#define	SWAPLONGLONG(y) \
(((unsigned long long)SWAPLONG((unsigned long)(y)) << 32) | (SWAPLONG((unsigned long)((y) >> 32))))
#endif /* __APPLE__ */

extern int dlt_to_linktype(int dlt);

extern int linktype_to_dlt(int linktype);

extern void swap_pseudo_headers(int linktype, struct pcap_pkthdr *hdr,
    u_char *data);

extern u_int max_snaplen_for_dlt(int dlt);
