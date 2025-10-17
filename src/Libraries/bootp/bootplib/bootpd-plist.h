/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
#ifndef _S_BOOTPD_PLIST_H
#define _S_BOOTPD_PLIST_H

/* 
 * bootpd-plist.h
 * - format of /etc/bootpd.plist
 */
/* 
 * Modification History
 *
 * June 30, 2006	Dieter Siegmund (dieter@apple.com)
 * - created
 */

/*
 * /etc/bootpd.plist is an xml plist.  The structure is:
 * <dict>
 *	detect_other_dhcp_server
 *	bootp_enabled
 *	dhcp_enabled
 *	old_netboot_enabled
 *	netboot_enabled
 *	relay_enabled
 *	allow
 *	deny
 *	relay_ip_list
 *	reply_threshold_seconds
 *	use_open_directory
 *	NetBoot <dict>
 *		shadow_size_meg
 *		afp_users_max
 *		age_time_seconds
 *		afp_uid_start	
 *	Subnets <array>
 *		[0] <dict>
 *			name
 *			net_address
 *			net_mask
 *			net_range
 *			supernet
 *			allocate
 *			lease_max
 *			lease_min
 *			dhcp_* (e.g. dhcp_router)
 */

/*
 * Encoding: (root) <dict>
 *
 * --------------------------+----------------------------------------------
 * Property		     | Type
 * --------------------------+----------------------------------------------
 * detect_other_dhcp_server  | <boolean>, <integer>, <string>
 * --------------------------+----------------------------------------------
 * bootp_enabled	     | <boolean>, 
 * dhcp_enabled,	     | <string>,
 * old_netboot_enabled,	     | <array> of <string> 
 * netboot_enabled,	     |
 * relay_enabled 	     |
 * --------------------------+----------------------------------------------
 * allow, deny,		     | <array> of <string>
 * relay_ip_list	     |
 * --------------------------+----------------------------------------------
 * reply_threshold_seconds   | <integer>, <string>
 * --------------------------+----------------------------------------------
 * use_open_directory        | <boolean>, <integer>, <string>
 * --------------------------+----------------------------------------------
 */

/*
 * Encoding: NetBoot <dict>
 *
 * --------------------------+----------------------------------------------
 * Property		     | Encoding	
 * --------------------------+----------------------------------------------
 * shadow_size_meg	     | <integer>, <string>
 * afp_users_max	     |
 * age_time_seconds	     |
 * afp_uid_start	     |
 * --------------------------+----------------------------------------------
 * machine_name_format	     | <string>
 * --------------------------+----------------------------------------------
 */

/*
 * Encoding: Subnets <array> of <dict>
 *
 * <dict> contains:
 * --------------------------+----------------------------------------------
 * Property		     | Encoding	
 * --------------------------+----------------------------------------------
 * name  		     | <string>
 * net_address		     |
 * net_mask		     |
 * supernet		     |
 * --------------------------+----------------------------------------------
 * net_range		     | <array> of <string>
 * --------------------------+----------------------------------------------
 * allocate		     | <boolean>
 * --------------------------+----------------------------------------------
 * lease_min, lease_max	     | <integer>, <string>
 * --------------------------+----------------------------------------------
 * dhcp_*		     | convert using dhcp option conversion table
 * --------------------------+----------------------------------------------
 */

#include <CoreFoundation/CFString.h>

#define BOOTPD_PLIST_NETBOOT	CFSTR("NetBoot")
#define BOOTPD_PLIST_SUBNETS	CFSTR("Subnets")

#endif /* _S_BOOTPD_PLIST_H */
