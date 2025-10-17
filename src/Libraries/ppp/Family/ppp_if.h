/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#ifndef _PPP_IF_H_
#define _PPP_IF_H_

/*
 * Network protocols we support.
 */
#define NP_IP	0		/* Internet Protocol V4 */
#define NP_IPV6	1		/* Internet Protocol V6 */
//#define NP_IPX	2		/* IPX protocol */
//#define NP_AT	3		/* Appletalk protocol */
#define NUM_NP	2		/* Number of NPs. */

/*
 * State of the interface.
 */
#define PPP_IF_STATE_DETACHING	1

struct ppp_if {
    /* first, the ifnet structure... */
    ifnet_t				net;		/* network-visible interface */

    /* administrative info */
    TAILQ_ENTRY(ppp_if) next;
    void				*host;		/* first client structure */
    u_int8_t			nbclients;	/* nb clients attached */
	u_int8_t			state;		/* state of the interface */
	lck_mtx_t			*mtx;		/* interface mutex */
	u_short				unit;		/* unit number (same as in ifnet_t) */
	
    /* ppp data */
    u_int16_t			mru;		/* max receive unit */
    TAILQ_HEAD(, ppp_link)  link_head; 	/* list of links attached to this interface */
    u_int8_t			nblinks;	/* # links currently attached */
    mbuf_t				outm;		/* mbuf currently being output */
    time_t				last_xmit; 	/* last proto packet sent on this interface */
    time_t				last_recv; 	/* last proto packet received on this interface */
    u_int32_t			sc_flags;	/* ppp private flags */
    struct slcompress	*vjcomp; 	/* vjc control buffer */
    enum NPmode			npmode[NUM_NP];	/* what to do with each net proto */
    enum NPAFmode		npafmode[NUM_NP];/* address filtering for each net proto */
	struct pppqueue		sndq;		/* send queue */
	bpf_packet_func		bpf_input;	/* bpf input function */
	bpf_packet_func		bpf_output;	/* bpf output function */
	
    /* data compression */
    void				*xc_state;	/* send compressor state */
    struct ppp_comp		*xcomp;		/* send compressor structure */
    void				*rc_state;	/* send compressor state */
    struct ppp_comp		*rcomp;		/* send compressor structure */

	/* network protocols data */
    int					ip_attached;
    struct in_addr		ip_src;
    struct in_addr		ip_dst;
    int					ipv6_attached;
    ifnet_t				lo_ifp;		/* loopback interface */
};


/*
 * Bits in sc_flags: SC_NO_TCP_CCID, SC_CCP_OPEN, SC_CCP_UP, SC_LOOP_TRAFFIC,
 * SC_MULTILINK, SC_MP_SHORTSEQ, SC_MP_XSHORTSEQ, SC_COMP_TCP, SC_REJ_COMP_TCP.
 */
#define SC_FLAG_BITS	(SC_NO_TCP_CCID|SC_CCP_OPEN|SC_CCP_UP|SC_LOOP_TRAFFIC \
			 |SC_MULTILINK|SC_MP_SHORTSEQ|SC_MP_XSHORTSEQ \
			 |SC_COMP_TCP|SC_REJ_COMP_TCP)


int ppp_if_init(void);
int ppp_if_dispose(void);
int ppp_if_attach(u_short *unit);
int ppp_if_attachclient(u_short unit, void *host, ifnet_t *ifp);
void ppp_if_detachclient(ifnet_t ifp, void *host);

int ppp_if_input(ifnet_t ifp, mbuf_t m, u_int16_t proto, u_int16_t hdrlen);
int ppp_if_control(ifnet_t ifp, u_long cmd, void *data);
int ppp_if_attachlink(struct ppp_link *link, int unit);
int ppp_if_detachlink(struct ppp_link *link);
int ppp_if_send(ifnet_t ifp, mbuf_t m);
void ppp_if_error(ifnet_t ifp);
int ppp_if_xmit(ifnet_t ifp, mbuf_t m);

bool ppp_if_host_has_unit(void *host);

#define APPLE_PPP_NAME	"ppp"



#endif /* _PPP_IF_H_ */
