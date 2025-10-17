/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
/*
 * DHCPv6Client.h
 * - API's to instantiate and interact with DHCPv6Client
 */

/* 
 * Modification History
 *
 * September 22, 2009		Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _S_DHCPV6CLIENT_H
#define _S_DHCPV6CLIENT_H

#include <stdint.h>
#include <stdbool.h>

#include "DHCPDUID.h"
#include "interfaces.h"

typedef struct DHCPv6Client * DHCPv6ClientRef;

typedef enum {
    kDHCPv6ClientNotificationTypeStatusChanged = 1,
    kDHCPv6ClientNotificationTypeGenerateSymptom = 2
} DHCPv6ClientNotificationType;

typedef
enum {
      kDHCPv6ClientModeNone = 0,
      kDHCPv6ClientModeStateless = 1,
      kDHCPv6ClientModeStatefulAddress = 2,
      kDHCPv6ClientModeStatefulPrefix = 3,
} DHCPv6ClientMode;

#include "ipconfigd_threads.h"

typedef void
(*DHCPv6ClientNotificationCallBack)(DHCPv6ClientRef client,
				    void * callback_arg,
				    DHCPv6ClientNotificationType type);

void
DHCPv6ClientSetRequestedOptions(uint16_t * requested_options,
				int requested_options_count);
bool
DHCPv6ClientOptionIsOK(int option);

DHCPv6ClientRef
DHCPv6ClientCreate(ServiceRef service_p);

DHCPv6ClientMode
DHCPv6ClientGetMode(DHCPv6ClientRef client);

void
DHCPv6ClientSetMode(DHCPv6ClientRef client, DHCPv6ClientMode mode);

void
DHCPv6ClientSetUsePrivateAddress(DHCPv6ClientRef client,
				 bool use_private_address);

void
DHCPv6ClientStart(DHCPv6ClientRef client);

void
DHCPv6ClientStop(DHCPv6ClientRef client);

void
DHCPv6ClientDiscardInformation(DHCPv6ClientRef client);

void
DHCPv6ClientRelease(DHCPv6ClientRef * client_p);

bool
DHCPv6ClientGetInfo(DHCPv6ClientRef client, ipv6_info_t * info_p);

void
DHCPv6ClientCopyAddresses(DHCPv6ClientRef client, 
			  inet6_addrlist_t * addr_list_p);

void
DHCPv6ClientHandleEvent(DHCPv6ClientRef client, IFEventID_t event_ID,
			void * event_data);

void
DHCPv6ClientSetNotificationCallBack(DHCPv6ClientRef client, 
				    DHCPv6ClientNotificationCallBack callback,
				    void * callback_arg);

bool
DHCPv6ClientIsActive(DHCPv6ClientRef client);

bool
DHCPv6ClientHasDNS(DHCPv6ClientRef client, bool * search_available);

void
DHCPv6ClientProvideSummary(DHCPv6ClientRef client,
			   CFMutableDictionaryRef summary);

#endif /* _S_DHCPV6CLIENT_H */
