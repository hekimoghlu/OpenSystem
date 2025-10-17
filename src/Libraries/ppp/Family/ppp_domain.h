/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
#ifndef __PPP_DOMAIN_H__
#define __PPP_DOMAIN_H__


/* ppp_domain is self contained */
#include <sys/sysctl.h>
#include "ppp_defs.h"
#include "if_ppplink.h"
#include "if_ppp.h"


#define PPPPROTO_CTL		1		/* control protocol for ifnet layer */

#define PPP_NAME		"PPP"		/* ppp family name */


struct sockaddr_ppp {
    u_int8_t	ppp_len;			/* sizeof(struct sockaddr_ppp) + variable part */
    u_int8_t	ppp_family;			/* AF_PPPCTL */
    u_int16_t	ppp_proto;			/* protocol coding address */
    u_int32_t 	ppp_cookie;			/* one long for protocol with few info */
    // variable len, the following are protocol specific addresses
};


struct ppp_link_event_data {
     u_int16_t          lk_index;
     u_int16_t          lk_unit;
     char               lk_name[IFNAMSIZ];
};

/* Define PPP events, as subclass of NETWORK_CLASS events */

#define KEV_PPP_NET_SUBCLASS 	3
#define KEV_PPP_LINK_SUBCLASS 	4



#ifdef KERNEL

#include <IOKit/IOLib.h>

int ppp_domain_init(void);
int ppp_domain_dispose(void);
int ppp_proto_add(void);
int ppp_proto_remove(void);

int ppp_proto_input(void *data, mbuf_t m);
void ppp_proto_free(void *data);

SYSCTL_DECL(_net_ppp);

/* Logs facilities */

#define LOGDBG(ifp, text) \
    if (ifnet_flags(ifp) & IFF_DEBUG) {	\
        IOLog text; 		\
    }

#define LOGRETURN(err, ret, text) \
    if (err) {			\
        IOLog(text, err); \
        return ret;		\
    }
	
#define LOGGOTOFAIL(err, text) \
    if (err) {			\
        IOLog(text, err); \
        goto fail;		\
    }

#define LOGNULLFAIL(ret, text) \
    if (ret == 0) {			\
        IOLog(text); \
        goto fail;		\
    }

#ifdef LOGDATA
#define LOGMBUF(text, m)   {		\
    short i;				\
    char *p = mtod((m), u_char *);	\
    IOLog(text);			\
    IOLog(" : 0x ");		\
    for (i = 0; i < (m)->m_len; i++)	\
       IOLog("%x ", p[i]);	\
    IOLog("\n");			\
}
#else
#define LOGMBUF(text, m)
#endif



/*
 * PPP queues.
 */
struct	pppqueue {
	mbuf_t head;
	mbuf_t tail;
	int	len;
	int	maxlen;
	int	drops;
};

int ppp_qfull(struct pppqueue *pppq);
void ppp_drop(struct pppqueue *pppq);
void ppp_enqueue(struct pppqueue *pppq, mbuf_t m);
mbuf_t ppp_dequeue(struct pppqueue *pppq);
void ppp_prepend(struct pppqueue *pppq, mbuf_t m);

#endif

#endif
