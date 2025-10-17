/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>

#include "pcap-int.h"

static char nosup[] = "live packet capture not supported on this system";

pcap_t *
pcap_create_interface(const char *device _U_, char *ebuf)
{
	(void)pcap_strlcpy(ebuf, nosup, PCAP_ERRBUF_SIZE);
	return (NULL);
}

int
pcap_platform_finddevs(pcap_if_list_t *devlistp _U_, char *errbuf _U_)
{
	/*
	 * There are no interfaces on which we can capture.
	 */
	return (0);
}

#ifdef _WIN32
int
pcap_lookupnet(const char *device _U_, bpf_u_int32 *netp _U_,
    bpf_u_int32 *maskp _U_, char *errbuf)
{
	(void)pcap_strlcpy(errbuf, nosup, PCAP_ERRBUF_SIZE);
	return (-1);
}
#endif

/*
 * Libpcap version string.
 */
const char *
pcap_lib_version(void)
{
	return (PCAP_VERSION_STRING);
}
