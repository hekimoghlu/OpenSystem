/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
#ifndef __IPSEC_UTILS_H__
#define __IPSEC_UTILS_H__

#import <net/if_var.h>
#import <netdb.h>
#define MAX_ADDRESS_STRLEN (sizeof("0000:0000:0000:0000:0000:0000:0000:0000%") + IFNAMSIZ)

#include <sys/kern_event.h>
#include "scnc_main.h"

/* IKE Configuration */
void IPSecConfigureVerboseLogging(CFMutableDictionaryRef ipsec_dict, int verbose_logging);
int IPSecValidateConfiguration(CFDictionaryRef ipsec_dict, char **error_text);
int IPSecApplyConfiguration(CFDictionaryRef ipsec_dict, char **error_text);
int IPSecRemoveConfiguration(CFDictionaryRef ipsec_dict, char **error_text);
int IPSecRemoveConfigurationFile(CFDictionaryRef ipsec_dict, char **error_text);
int IPSecKickConfiguration(void);

int IPSecSelfRepair(void);
int IPSecFlushAll(void);

/* Kernel Policies */
int IPSecCountPolicies(CFDictionaryRef ipsec_dict);
int IPSecInstallPolicies(CFDictionaryRef ipsec_dict, CFIndex index, char **error_text);
int IPSecRemovePolicies(CFDictionaryRef ipsec_dict, CFIndex index, char **error_text);
int IPSecInstallRoutes(struct service *serv, CFDictionaryRef ipsec_dict, CFIndex index, char **error_text, struct in_addr gateway);
int IPSecRemoveRoutes(struct service *serv, CFDictionaryRef ipsec_dict, CFIndex index, char **error_text, struct in_addr gateway);

/* Kernel Security Associations */
int IPSecRemoveSecurityAssociations(struct sockaddr *src, struct sockaddr *dst);
int IPSecSetSecurityAssociationsPreference(int *oldval, int newval);

/* Functions to manipulate well known configurations */
CFMutableDictionaryRef 
IPSecCreateL2TPDefaultConfiguration(struct sockaddr *src, struct sockaddr *dst, char *dst_hostName, 
		CFStringRef authenticationMethod, uint hasLocalIdentifier, int isClient, int natt_multiple_users, CFStringRef identifierVerification, int dhgroup2_aggressive);
CFMutableDictionaryRef 
IPSecCreateCiscoDefaultConfiguration(struct sockaddr *src, struct sockaddr_in *dst, CFStringRef dst_hostName, 
		CFStringRef authenticationMethod, uint hasLocalIdentifier, int isClient, int natt_multiple_users, CFStringRef identifierVerification, int dhgroup2_aggressive);
int IPSecIsAggressiveMode(CFStringRef authenticationMethod, int hasLocalIdentifier, int isClient);

/* Miscellaneous */
int get_src_address(struct sockaddr *src, const struct sockaddr *dst, char *ifscope, char *if_name);
u_int32_t get_if_media(char *if_name);
u_int32_t get_if_baudrate(char *if_name);
u_int32_t get_if_mtu(char *if_name);

void IPSecLogVPNInterfaceAddressEvent (const char                  *location,
									   struct kern_event_msg *ev_msg,
									   int                    wait_interface_timeout,
								 	   char                  *interface,
									   struct in_addr        *our_address);

void update_service_route (struct service	*serv,
						   in_addr_t	local_addr,
						   in_addr_t	local_mask,
						   in_addr_t	dest_addr,
						   in_addr_t	dest_mask,
						   in_addr_t	gtwy_addr,
						   uint16_t			flags,
						   int				installed);
void free_service_routes (struct service	*serv);

/* creates a directory path from string */
int makepath( char *path);

int racoon_validate_cfg_str (char *str_buf);

/* return the pid of racoon process */
pid_t racoon_pid(void);

#endif
