/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#ifndef _S_IPCONFIGD_GLOBALS_H
#define _S_IPCONFIGD_GLOBALS_H

/*
 * ipconfigd_globals.h
 * - ipconfigd global definitions
 */
/* 
 * Modification History
 *
 * May 22, 2000		Dieter Siegmund (dieter@apple.com)
 * - created
 */

#include <CoreFoundation/CFString.h>
#include "mylog.h"
#include "util.h"
#include <sys/stat.h>
#include "IPConfigurationAgentUtil.h"

#define IPCONFIGURATION_PRIVATE_DIR	"/var/db/dhcpclient"
#define DHCPCLIENT_LEASES_DIR		IPCONFIGURATION_PRIVATE_DIR "/leases"
#define ARP_PROBE_FAILURE_RETRY_TIME	(8.0)

void
remove_unused_ip(const char * ifname, struct in_addr ip);

INLINE void
ipconfigd_create_paths(void)
{
    if (create_path(DHCPCLIENT_LEASES_DIR, 0700) < 0) {
	my_log(LOG_ERR, "failed to create " 
	       DHCPCLIENT_LEASES_DIR ", %s (%d)", strerror(errno), errno);
	return;
    }
    return;

}

#endif /* _S_IPCONFIGD_GLOBALS_H */
