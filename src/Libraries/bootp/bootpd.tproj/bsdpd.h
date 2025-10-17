/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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


#ifndef _S_BSDPD_H
#define _S_BSDPD_H

/*
 * Copyright (c) 1999-2002 Apple Inc. All rights reserved.
 *
 * @APPLE_LICENSE_HEADER_START@
 * 
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 * 
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 * 
 * @APPLE_LICENSE_HEADER_END@
 */

/*
 * bsdpd.h
 * - Boot Server Discovery Protocol (BSDP) server definitions
 */

/*
 * Modification History
 *
 * Dieter Siegmund (dieter@apple.com)		November 23, 1999
 * - created
 */

#include "bsdp.h"
#include "nbsp.h"
#include "bootpd.h"
#include <CoreFoundation/CFDictionary.h>

#define ROOT_UID		0

#define SHADOW_SIZE_DEFAULT	48

#define OLD_NETBOOT_SYSID		"/NetBoot1"

boolean_t
bsdp_init(CFDictionaryRef plist);

boolean_t
is_bsdp_packet(dhcpol_t * rq_options, char * arch, char * sysid,
	       dhcpol_t * rq_vsopt, bsdp_version_t * client_version,
	       boolean_t * is_old_netboot);

void
bsdp_request(request_t * request, dhcp_msgtype_t dhcpmsg,
	     const char * arch, const char * sysid, dhcpol_t * rq_vsopt,
	     bsdp_version_t client_version, boolean_t is_old_netboot);

boolean_t
old_netboot_request(request_t * request);

void
bsdp_dhcp_request(request_t * request, dhcp_msgtype_t dhcpmsg);

/**
 ** Globals
 **/
extern uint32_t		G_age_time_seconds;
extern gid_t		G_admin_gid;
extern boolean_t	G_disk_space_warned; 
extern uint32_t		G_shadow_size_meg;
extern NBSPListRef	G_client_sharepoints;

#endif /* _S_BSDPD_H */
