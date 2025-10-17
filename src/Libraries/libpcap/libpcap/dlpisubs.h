/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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

#ifndef dlpisubs_h
#define	dlpisubs_h

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Private data for capturing on DLPI devices.
 */
struct pcap_dlpi {
#ifdef HAVE_LIBDLPI
	dlpi_handle_t dlpi_hd;
#endif /* HAVE_LIBDLPI */
#ifdef DL_HP_RAWDLS
	int send_fd;
#endif /* DL_HP_RAWDLS */

	struct pcap_stat stat;
};

/*
 * Functions defined by dlpisubs.c.
 */
int pcap_stats_dlpi(pcap_t *, struct pcap_stat *);
int pcap_process_pkts(pcap_t *, pcap_handler, u_char *, int, u_char *, int);
int pcap_process_mactype(pcap_t *, u_int);
#ifdef HAVE_SYS_BUFMOD_H
int pcap_conf_bufmod(pcap_t *, int);
#endif
int pcap_alloc_databuf(pcap_t *);
int strioctl(int, int, int, char *);

#ifdef __cplusplus
}
#endif

#endif
