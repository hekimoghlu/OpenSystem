/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#include "extract.h"

/*
 * Definition to let us compile most of the IPv6 code even on systems
 * without IPv6 support.
 */
#ifndef INET6_ADDRSTRLEN
#define INET6_ADDRSTRLEN	46
#endif

/* Name to address translation routines. */

enum {
    LINKADDR_ETHER,
    LINKADDR_FRELAY,
    LINKADDR_IEEE1394,
    LINKADDR_ATM,
    LINKADDR_OTHER
};

#define BUFSIZE 128

extern const char *linkaddr_string(netdissect_options *, const uint8_t *, const unsigned int, const unsigned int);
extern const char *etheraddr_string(netdissect_options *, const uint8_t *);
extern const char *le64addr_string(netdissect_options *, const uint8_t *);
extern const char *tcpport_string(netdissect_options *, u_short);
extern const char *udpport_string(netdissect_options *, u_short);
extern const char *isonsap_string(netdissect_options *, const uint8_t *, u_int);
extern const char *dnaddr_string(netdissect_options *, u_short);
extern const char *ipxsap_string(netdissect_options *, u_short);
extern const char *ipaddr_string(netdissect_options *, const u_char *);
extern const char *ip6addr_string(netdissect_options *, const u_char *);
extern const char *intoa(uint32_t);

extern void init_addrtoname(netdissect_options *, uint32_t, uint32_t);
extern struct hnamemem *newhnamemem(netdissect_options *);
extern struct h6namemem *newh6namemem(netdissect_options *);
extern const char * ieee8021q_tci_string(const uint16_t);

/* macro(s) and inline function(s) with setjmp/longjmp logic to call
 * the X_string() function(s) after bounds checking.
 * The macro(s) must be used on a packet buffer pointer.
 */

static inline const char *
get_linkaddr_string(netdissect_options *ndo, const uint8_t *p,
    const unsigned int type, const unsigned int len)
{
        if (!ND_TTEST_LEN(p, len))
                nd_trunc_longjmp(ndo);
        return linkaddr_string(ndo, p, type, len);
}

static inline const char *
get_etheraddr_string(netdissect_options *ndo, const uint8_t *p)
{
        if (!ND_TTEST_LEN(p, MAC_ADDR_LEN))
                nd_trunc_longjmp(ndo);
        return etheraddr_string(ndo, p);
}

static inline const char *
get_le64addr_string(netdissect_options *ndo, const u_char *p)
{
        if (!ND_TTEST_8(p))
                nd_trunc_longjmp(ndo);
        return le64addr_string(ndo, p);
}

static inline const char *
get_isonsap_string(netdissect_options *ndo, const uint8_t *nsap,
    u_int nsap_length)
{
	if (!ND_TTEST_LEN(nsap, nsap_length))
                nd_trunc_longjmp(ndo);
        return isonsap_string(ndo, nsap, nsap_length);
}

static inline const char *
get_ipaddr_string(netdissect_options *ndo, const u_char *p)
{
        if (!ND_TTEST_4(p))
                nd_trunc_longjmp(ndo);
        return ipaddr_string(ndo, p);
}

static inline const char *
get_ip6addr_string(netdissect_options *ndo, const u_char *p)
{
        if (!ND_TTEST_16(p))
                nd_trunc_longjmp(ndo);
        return ip6addr_string(ndo, p);
}

#define GET_LINKADDR_STRING(p, type, len) get_linkaddr_string(ndo, (const u_char *)(p), type, len)
#define GET_ETHERADDR_STRING(p) get_etheraddr_string(ndo, (const u_char *)(p))
#define GET_LE64ADDR_STRING(p) get_le64addr_string(ndo, (const u_char *)(p))
#define GET_ISONSAP_STRING(nsap, nsap_length) get_isonsap_string(ndo, (const u_char *)(nsap), nsap_length)
#define GET_IPADDR_STRING(p) get_ipaddr_string(ndo, (const u_char *)(p))
#define GET_IP6ADDR_STRING(p) get_ip6addr_string(ndo, (const u_char *)(p))
