/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#include <sys/cdefs.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <net/route.h>
#include <errno.h>
#include <ifaddrs.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define	SALIGN	(sizeof(uint32_t) - 1)
#define	SA_RLEN(sa)	((sa)->sa_len ? (((sa)->sa_len + SALIGN) & ~SALIGN) : \
(SALIGN + 1))
#define	MAX_SYSCTL_TRY	5
#define	RTA_MASKS	(RTA_GATEWAY | RTA_IFP | RTA_IFA)
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
 *//* We can use NET_RT_IFLIST2 and RTM_NEWMADDR2 on Darwin */
#define DARWIN_COMPAT

#ifdef DARWIN_COMPAT
#define GIM_SYSCTL_MIB NET_RT_IFLIST2
#define GIM_RTM_ADDR RTM_NEWMADDR2
#else
#define GIM_SYSCTL_MIB NET_RT_IFMALIST
#define GIM_RTM_ADDR RTM_NEWMADDR
#endif

int
getifmaddrs(struct ifmaddrs **pif)
{
	int icnt = 1;
	int dcnt = 0;
	int ntry = 0;
	size_t len;
	size_t needed;
	int mib[6];
	int i;
	char *buf;
	char *data;
	char *next;
	char *p;
	struct ifma_msghdr2 *ifmam;
	struct ifmaddrs *ifa, *ift;
	struct rt_msghdr *rtm;
	struct sockaddr *sa;

	mib[0] = CTL_NET;
	mib[1] = PF_ROUTE;
	mib[2] = 0;             /* protocol */
	mib[3] = 0;             /* wildcard address family */
	mib[4] = GIM_SYSCTL_MIB;
	mib[5] = 0;             /* no flags */
	do {
		if (sysctl(mib, 6, NULL, &needed, NULL, 0) < 0)
			return (-1);
		if ((buf = malloc(needed)) == NULL)
			return (-1);
		if (sysctl(mib, 6, buf, &needed, NULL, 0) < 0) {
			if (errno != ENOMEM || ++ntry >= MAX_SYSCTL_TRY) {
				free(buf);
				return (-1);
			}
			free(buf);
			buf = NULL;
		} 
	} while (buf == NULL);

	for (next = buf; next < buf + needed; next += rtm->rtm_msglen) {
		rtm = (struct rt_msghdr *)(void *)next;
		if (rtm->rtm_version != RTM_VERSION)
			continue;
		switch (rtm->rtm_type) {
			case GIM_RTM_ADDR:
				ifmam = (struct ifma_msghdr2 *)(void *)rtm;
				if ((ifmam->ifmam_addrs & RTA_IFA) == 0)
					break;
				icnt++;
				p = (char *)(ifmam + 1);
				for (i = 0; i < RTAX_MAX; i++) {
					if ((RTA_MASKS & ifmam->ifmam_addrs &
						 (1 << i)) == 0)
						continue;
					sa = (struct sockaddr *)(void *)p;
					len = SA_RLEN(sa);
					dcnt += len;
					p += len;
				}
				break;
		}
	}

	data = malloc(sizeof(struct ifmaddrs) * icnt + dcnt);
	if (data == NULL) {
		free(buf);
		return (-1);
	}

	ifa = (struct ifmaddrs *)(void *)data;
	data += sizeof(struct ifmaddrs) * icnt;

	memset(ifa, 0, sizeof(struct ifmaddrs) * icnt);
	ift = ifa;

	for (next = buf; next < buf + needed; next += rtm->rtm_msglen) {
		rtm = (struct rt_msghdr *)(void *)next;
		if (rtm->rtm_version != RTM_VERSION)
			continue;

		switch (rtm->rtm_type) {
			case GIM_RTM_ADDR:
				ifmam = (struct ifma_msghdr2 *)(void *)rtm;
				if ((ifmam->ifmam_addrs & RTA_IFA) == 0)
					break;

				p = (char *)(ifmam + 1);
				for (i = 0; i < RTAX_MAX; i++) {
					if ((RTA_MASKS & ifmam->ifmam_addrs &
						 (1 << i)) == 0)
						continue;
					sa = (struct sockaddr *)(void *)p;
					len = SA_RLEN(sa);
					switch (i) {
						case RTAX_GATEWAY:
							ift->ifma_lladdr =
							(struct sockaddr *)(void *)data;
							memcpy(data, p, len);
							data += len;
							break;

						case RTAX_IFP:
							ift->ifma_name =
							(struct sockaddr *)(void *)data;
							memcpy(data, p, len);
							data += len;
							break;

						case RTAX_IFA:
							ift->ifma_addr =
							(struct sockaddr *)(void *)data;
							memcpy(data, p, len);
							data += len;
							break;

						default:
							data += len;
							break;
					}
					p += len;
				}
				ift->ifma_next = ift + 1;
				ift = ift->ifma_next;
				break;
		}
	}

	free(buf);

	if (ift > ifa) {
		ift--;
		ift->ifma_next = NULL;
		*pif = ifa;
	} else {
		*pif = NULL;
		free(ifa);
	}
	return (0);
}

void
freeifmaddrs(struct ifmaddrs *ifmp)
{
	free(ifmp);
}

