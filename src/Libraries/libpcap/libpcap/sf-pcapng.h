/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#ifndef sf_pcapng_h
#define	sf_pcapng_h

#ifdef __APPLE__
extern pcap_t *pcap_ng_check_header(const uint8_t *magic, FILE *fp,
    u_int precision, char *errbuf, int *err, int isng);

struct block_cursor;
void *
get_from_block_data(struct block_cursor *cursor, size_t chunk_size,
		    char *errbuf);
#else
extern pcap_t *pcap_ng_check_header(const uint8_t *magic, FILE *fp,
    u_int precision, char *errbuf, int *err);
#endif /* __APPLE__ */

#endif
