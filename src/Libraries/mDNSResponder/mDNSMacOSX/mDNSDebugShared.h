/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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
#ifndef MDNS_DEBUG_SHARED_H
#define MDNS_DEBUG_SHARED_H

#include <mdns/general.h>

MDNS_CLOSED_OPTIONS(mDNSNetworkChangeEventFlags_t, uint32_t,
	mDNSNetworkChangeEventFlag_None				= 0,
	mDNSNetworkChangeEventFlag_LocalHostname	= (1U << 0),
	mDNSNetworkChangeEventFlag_ComputerName		= (1U << 1),
	mDNSNetworkChangeEventFlag_DNS				= (1U << 2),
	mDNSNetworkChangeEventFlag_DynamicDNS		= (1U << 3),
	mDNSNetworkChangeEventFlag_IPv4LL			= (1U << 4),
	mDNSNetworkChangeEventFlag_P2PLike			= (1U << 5),
);

#endif	// MDNS_DEBUG_SHARED_H
