/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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
#ifndef _NATTRAVERSAL_H
#define _NATTRAVERSAL_H

#include "vendorid.h"
#ifdef ENABLE_NATT
#ifdef ENABLE_FRAG
#include "isakmp_frag.h"
#endif /* ENABLE_NATT */
#endif /* ENABLE_FRAG */

#define UDP_ENCAP_ESPINUDP	2	/* to make it compile - we don't use this */

#define	NAT_ANNOUNCED		(1L<<0)
#define	NAT_DETECTED_ME		(1L<<1)
#define	NAT_DETECTED_PEER	(1L<<2)
#define	NAT_PORTS_CHANGED	(1L<<3)
#define	NAT_KA_QUEUED		(1L<<4)
#define	NAT_ADD_NON_ESP_MARKER	(1L<<5)

#define	NATT_AVAILABLE(iph1)	((iph1)->natt_flags & NAT_ANNOUNCED)

#define	NAT_DETECTED	(NAT_DETECTED_ME | NAT_DETECTED_PEER)

#define	NON_ESP_MARKER_LEN	sizeof(u_int32_t)
#define	NON_ESP_MARKER_USE(iph1)	((iph1)->natt_flags & NAT_ADD_NON_ESP_MARKER)

#ifdef ENABLE_NATT
#ifdef ENABLE_FRAG
#define PH1_NON_ESP_EXTRA_LEN(iph1, sendbuf) ((iph1->frag && sendbuf->l > ISAKMP_FRAG_MAXLEN) ? 0: (NON_ESP_MARKER_USE(iph1) ? NON_ESP_MARKER_LEN : 0))
#define PH2_NON_ESP_EXTRA_LEN(iph2, sendbuf) ((iph2->ph1->frag && sendbuf->l > ISAKMP_FRAG_MAXLEN) ? 0: (NON_ESP_MARKER_USE(iph2->ph1) ? NON_ESP_MARKER_LEN : 0))
#define PH1_FRAG_FLAGS(iph1) (NON_ESP_MARKER_USE(iph1) ? FRAG_PUT_NON_ESP_MARKER : 0)
#define PH2_FRAG_FLAGS(iph2) (NON_ESP_MARKER_USE(iph2->ph1) ? FRAG_PUT_NON_ESP_MARKER : 0)
#else
#define PH1_NON_ESP_EXTRA_LEN(iph1, sendbuf) (NON_ESP_MARKER_USE(iph1) ? NON_ESP_MARKER_LEN : 0)
#define PH2_NON_ESP_EXTRA_LEN(iph2, sendbuf) (NON_ESP_MARKER_USE(iph2->ph1) ? NON_ESP_MARKER_LEN : 0)
#define PH1_FRAG_FLAGS(iph1) 0
#define PH2_FRAG_FLAGS(iph2) 0
#endif
#else
#define PH1_NON_ESP_EXTRA_LEN(iph1, sendbuf) 0
#define PH2_NON_ESP_EXTRA_LEN(iph2, sendbuf) 0
#define PH1_FRAG_FLAGS(iph1) 0
#define PH2_FRAG_FLAGS(iph2) 0
#endif

/* These are the values from parsing "remote {}" 
   block of the config file. */
#define NATT_OFF	FALSE	/* = 0 */
#define NATT_ON		TRUE	/* = 1 */
#define NATT_FORCE	2

struct ph1natt_options {
  int		version;
  u_int16_t	float_port;
  u_int16_t	mode_udp_tunnel;
  u_int16_t	mode_udp_transport;
  u_int16_t	encaps_type; /* ESPINUDP / ESPINUDP_NON_IKE */
  u_int16_t	mode_udp_diff;
  u_int16_t	payload_nat_d;
  u_int16_t	payload_nat_oa;
};

struct ph2natt {
  u_int8_t	type;
  u_int16_t	sport;
  u_int16_t	dport;
  struct sockaddr_storage	*oa;
  u_int16_t	frag;
};

int natt_vendorid (int vid);
vchar_t *natt_hash_addr (phase1_handle_t *iph1, struct sockaddr_storage *addr);
int natt_compare_addr_hash (phase1_handle_t *iph1, vchar_t *natd_received, int natd_seq);
int natt_udp_encap (int encmode);
int natt_fill_options (struct ph1natt_options *opts, int version);
void natt_float_ports (phase1_handle_t *iph1);
void natt_handle_vendorid (phase1_handle_t *iph1, int vid_numeric);
int create_natoa_payloads(phase2_handle_t *iph2, vchar_t **, vchar_t **);
struct sockaddr_storage * process_natoa_payload(vchar_t *buf);

struct payload_list *
isakmp_plist_append_natt_vids (struct payload_list *plist, vchar_t *vid_natt[MAX_NATT_VID_COUNT]);

/* Walk through all rmconfigs and tell if NAT-T is enabled in at least one. */
int natt_enabled_in_rmconf (void);

#endif /* _NATTRAVERSAL_H */
