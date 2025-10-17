/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
#ifndef _S_GLOBALS_H
#define _S_GLOBALS_H
#include <mach/boolean.h>
#include <stdint.h>
#include "timer.h"
#include "IPConfigurationControlPrefs.h"
#include "DHCPDUID.h"

extern uint16_t 		G_client_port;
extern boolean_t		G_dhcp_accepts_bootp;
extern boolean_t		G_dhcp_failure_configures_linklocal;
extern boolean_t		G_dhcp_success_deconfigures_linklocal;
extern int			G_dhcp_init_reboot_retry_count;
extern int			G_dhcp_select_retry_count;
extern int			G_dhcp_allocate_linklocal_at_retry_count;
extern int			G_dhcp_generate_failure_symptom_at_retry_count;
extern int			G_dhcp_router_arp_at_retry_count;
extern uint16_t			G_server_port;
extern int			G_gather_secs;
extern int			G_initial_wait_secs;
extern int			G_max_wait_secs;
extern int			G_gather_secs;
extern int			G_link_inactive_secs;
extern int			G_max_retries;
extern boolean_t 		G_must_broadcast;
extern Boolean			G_IPConfiguration_verbose;
extern boolean_t		G_router_arp;
extern int			G_router_arp_wifi_lease_start_threshold_secs;
extern int			G_dhcp_defend_ip_address_interval_secs;
extern int			G_dhcp_defend_ip_address_count;
extern int			G_dhcp_lease_write_t1_threshold_secs;
extern int			G_manual_conflict_retry_interval_secs;
extern boolean_t		G_dhcpv6_enabled;
extern boolean_t		G_dhcpv6_stateful_enabled;
extern DHCPDUIDType		G_dhcp_duid_type;
extern boolean_t		G_is_netboot;
extern int			G_min_short_wake_interval_secs;
extern int			G_min_wake_interval_secs;
extern int			G_wake_skew_secs;

extern const unsigned char	G_rfc_magic[4];
extern const struct in_addr	G_ip_broadcast;
extern const struct in_addr	G_ip_zeroes;
extern IPConfigurationInterfaceTypes G_awd_interface_types;

#include "ipconfigd_globals.h"

#endif /* _S_GLOBALS_H */
