/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#include <sys/param.h>
#include <sys/socket.h>

#include <security/audit/audit.h>

#include <bsm/audit_domain.h>
#include <bsm/audit_record.h>

#if CONFIG_AUDIT
struct bsm_domain {
	u_short bd_bsm_domain;
	int     bd_local_domain;
};

#define PF_NO_LOCAL_MAPPING     -600

static const struct bsm_domain bsm_domains[] = {
	{ .bd_bsm_domain = BSM_PF_UNSPEC, .bd_local_domain = PF_UNSPEC },
	{ .bd_bsm_domain = BSM_PF_LOCAL, .bd_local_domain = PF_LOCAL },
	{ .bd_bsm_domain = BSM_PF_INET, .bd_local_domain = PF_INET },
	{ .bd_bsm_domain = BSM_PF_IMPLINK,
#ifdef PF_IMPLINK
	  .bd_local_domain = PF_IMPLINK
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_PUP,
#ifdef PF_PUP
	  .bd_local_domain = PF_PUP
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_CHAOS,
#ifdef PF_CHAOS
	  .bd_local_domain = PF_CHAOS
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_NS,
#ifdef PF_NS
	  .bd_local_domain = PF_NS
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_NBS,
#ifdef PF_NBS
	  .bd_local_domain = PF_NBS
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_ECMA,
#ifdef PF_ECMA
	  .bd_local_domain = PF_ECMA
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_DATAKIT,
#ifdef PF_DATAKIT
	  .bd_local_domain = PF_DATAKIT
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_CCITT,
#ifdef PF_CCITT
	  .bd_local_domain = PF_CCITT
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_SNA, .bd_local_domain = PF_SNA },
	{ .bd_bsm_domain = BSM_PF_DECnet, .bd_local_domain = PF_DECnet },
	{ .bd_bsm_domain = BSM_PF_DLI,
#ifdef PF_DLI
	  .bd_local_domain = PF_DLI
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_LAT,
#ifdef PF_LAT
	  .bd_local_domain = PF_LAT
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_HYLINK,
#ifdef PF_HYLINK
	  .bd_local_domain = PF_HYLINK
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_APPLETALK, .bd_local_domain = PF_APPLETALK },
	{ .bd_bsm_domain = BSM_PF_NIT,
#ifdef PF_NIT
	  .bd_local_domain = PF_NIT
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_802,
#ifdef PF_802
	  .bd_local_domain = PF_802
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_OSI,
#ifdef PF_OSI
	  .bd_local_domain = PF_OSI
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_X25,
#ifdef PF_X25
	  .bd_local_domain = PF_X25
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_OSINET,
#ifdef PF_OSINET
	  .bd_local_domain = PF_OSINET
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_GOSIP,
#ifdef PF_GOSIP
	  .bd_local_domain = PF_GOSIP
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_IPX, .bd_local_domain = PF_IPX },
	{ .bd_bsm_domain = BSM_PF_ROUTE, .bd_local_domain = PF_ROUTE },
	{ .bd_bsm_domain = BSM_PF_LINK,
#ifdef PF_LINK
	  .bd_local_domain = PF_LINK
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_INET6, .bd_local_domain = PF_INET6 },
	{ .bd_bsm_domain = BSM_PF_KEY, .bd_local_domain = PF_KEY },
	{ .bd_bsm_domain = BSM_PF_NCA,
#ifdef PF_NCA
	  .bd_local_domain = PF_NCA
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_POLICY,
#ifdef PF_POLICY
	  .bd_local_domain = PF_POLICY
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_INET_OFFLOAD,
#ifdef PF_INET_OFFLOAD
	  .bd_local_domain = PF_INET_OFFLOAD
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_NETBIOS,
#ifdef PF_NETBIOS
	  .bd_local_domain = PF_NETBIOS
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_ISO,
#ifdef PF_ISO
	  .bd_local_domain = PF_ISO
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_XTP,
#ifdef PF_XTP
	  .bd_local_domain = PF_XTP
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_COIP,
#ifdef PF_COIP
	  .bd_local_domain = PF_COIP
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_CNT,
#ifdef PF_CNT
	  .bd_local_domain = PF_CNT
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_RTIP,
#ifdef PF_RTIP
	  .bd_local_domain = PF_RTIP
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_SIP,
#ifdef PF_SIP
	  .bd_local_domain = PF_SIP
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_PIP,
#ifdef PF_PIP
	  .bd_local_domain = PF_PIP
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_ISDN,
#ifdef PF_ISDN
	  .bd_local_domain = PF_ISDN
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_E164,
#ifdef PF_E164
	  .bd_local_domain = PF_E164
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_NATM,
#ifdef PF_NATM
	  .bd_local_domain = PF_NATM
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_ATM,
#ifdef PF_ATM
	  .bd_local_domain = PF_ATM
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_NETGRAPH,
#ifdef PF_NETGRAPH
	  .bd_local_domain = PF_NETGRAPH
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_SLOW,
#ifdef PF_SLOW
	  .bd_local_domain = PF_SLOW
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_SCLUSTER,
#ifdef PF_SCLUSTER
	  .bd_local_domain = PF_SCLUSTER
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_ARP,
#ifdef PF_ARP
	  .bd_local_domain = PF_ARP
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_BLUETOOTH,
#ifdef PF_BLUETOOTH
	  .bd_local_domain = PF_BLUETOOTH
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_IEEE80211,
#ifdef PF_IEEE80211
	  .bd_local_domain = PF_IEEE80211
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_AX25,
#ifdef PF_AX25
	  .bd_local_domain = PF_AX25
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_ROSE,
#ifdef PF_ROSE
	  .bd_local_domain = PF_ROSE
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_NETBEUI,
#ifdef PF_NETBEUI
	  .bd_local_domain = PF_NETBEUI
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_SECURITY,
#ifdef PF_SECURITY
	  .bd_local_domain = PF_SECURITY
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_PACKET,
#ifdef PF_PACKET
	  .bd_local_domain = PF_PACKET
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_ASH,
#ifdef PF_ASH
	  .bd_local_domain = PF_ASH
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_ECONET,
#ifdef PF_ECONET
	  .bd_local_domain = PF_ECONET
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_ATMSVC,
#ifdef PF_ATMSVC
	  .bd_local_domain = PF_ATMSVC
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_IRDA,
#ifdef PF_IRDA
	  .bd_local_domain = PF_IRDA
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_PPPOX,
#ifdef PF_PPPOX
	  .bd_local_domain = PF_PPPOX
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_WANPIPE,
#ifdef PF_WANPIPE
	  .bd_local_domain = PF_WANPIPE
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_LLC,
#ifdef PF_LLC
	  .bd_local_domain = PF_LLC
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_CAN,
#ifdef PF_CAN
	  .bd_local_domain = PF_CAN
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_TIPC,
#ifdef PF_TIPC
	  .bd_local_domain = PF_TIPC
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_IUCV,
#ifdef PF_IUCV
	  .bd_local_domain = PF_IUCV
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_RXRPC,
#ifdef PF_RXRPC
	  .bd_local_domain = PF_RXRPC
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
	{ .bd_bsm_domain = BSM_PF_PHONET,
#ifdef PF_PHONET
	  .bd_local_domain = PF_PHONET
#else
	  .bd_local_domain = PF_NO_LOCAL_MAPPING
#endif
	},
};
static const int bsm_domains_count = sizeof(bsm_domains) /
    sizeof(bsm_domains[0]);

static const struct bsm_domain *
bsm_lookup_local_domain(int local_domain)
{
	int i;

	for (i = 0; i < bsm_domains_count; i++) {
		if (bsm_domains[i].bd_local_domain == local_domain) {
			return &bsm_domains[i];
		}
	}
	return NULL;
}

u_short
au_domain_to_bsm(int local_domain)
{
	const struct bsm_domain *bstp;

	bstp = bsm_lookup_local_domain(local_domain);
	if (bstp == NULL) {
		return BSM_PF_UNKNOWN;
	}
	return bstp->bd_bsm_domain;
}

static const struct bsm_domain *
bsm_lookup_bsm_domain(u_short bsm_domain)
{
	int i;

	for (i = 0; i < bsm_domains_count; i++) {
		if (bsm_domains[i].bd_bsm_domain == bsm_domain) {
			return &bsm_domains[i];
		}
	}
	return NULL;
}

int
au_bsm_to_domain(u_short bsm_domain, int *local_domainp)
{
	const struct bsm_domain *bstp;

	bstp = bsm_lookup_bsm_domain(bsm_domain);
	if (bstp == NULL || bstp->bd_local_domain) {
		return -1;
	}
	*local_domainp = bstp->bd_local_domain;
	return 0;
}
#endif /* CONFIG_AUDIT */
