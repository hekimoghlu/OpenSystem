/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
#ifndef _S_IPCONFIGD_H
#define _S_IPCONFIGD_H

#include "ipconfigd_types.h"

/**
 ** Routines in support of MiG interface
 **/
ipconfig_status_t
ipconfig_method_info_from_plist(CFPropertyListRef plist,
				ipconfig_method_info_t info);
int
get_if_count();

ipconfig_status_t
get_if_addr(const char * name, ip_address_t * addr);

ipconfig_status_t
get_if_option(const char * name, int option_code, 
	      dataOut_t * option_data,
	      mach_msg_type_number_t *option_dataCnt);

ipconfig_status_t
get_if_packet(const char * name, dataOut_t * packet,
	      mach_msg_type_number_t *packetCnt);

ipconfig_status_t
get_if_v6_packet(const char * name, dataOut_t * packet,
		 mach_msg_type_number_t *packetCnt);

ipconfig_status_t
set_if(const char * name, ipconfig_method_info_t info);

ipconfig_status_t
add_service(const char * name,
	    ipconfig_method_info_t info,
	    ServiceID service_id,
	    CFDictionaryRef plist, pid_t pid);

ipconfig_status_t
set_service(const char * name,
	    ipconfig_method_info_t info,
	    ServiceID service_id);

ipconfig_status_t
remove_service_with_id(const char * name,
		       ServiceID service_id);

ipconfig_status_t
find_service(const char * name,
	     boolean_t exact,
	     ipconfig_method_info_t info,
	     ServiceID service_id);

ipconfig_status_t
remove_service(const char * name,
	       ipconfig_method_info_t info);

ipconfig_status_t
refresh_service(const char * name,
		ServiceID service_id);

ipconfig_status_t
is_service_valid(const char * name, ServiceID service_id);

ipconfig_status_t
forget_network(const char * name, CFStringRef ssid);

ipconfig_status_t
get_if_ra(const char * name, xmlDataOut_t * ra_data,
	  mach_msg_type_number_t *ra_data_cnt);

ipconfig_status_t
copy_if_summary(const char * name, CFDictionaryRef * summary);

ipconfig_status_t
copy_interface_list(CFArrayRef * list);

ipconfig_status_t
get_dhcp_duid(dataOut_t * dhcp_duid, mach_msg_type_number_t * dhcp_duid_cnt);

ipconfig_status_t
get_dhcp_ia_id(const char * name, DHCPIAID * ia_id_p);

#endif /* _S_IPCONFIGD_H */
