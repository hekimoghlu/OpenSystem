/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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
 * wireless.h
 * - CF object to retrieve Wi-Fi information
 */
/* 
 * Modification History
 *
 * July 6, 2020 	Dieter Siegmund (dieter@apple.com)
 * - moved out of ipconfigd.c
 */
#ifndef _S_WIRELESS_H
#define _S_WIRELESS_H

#include <stdint.h>
#include <stdbool.h>
#include <CoreFoundation/CFString.h>
#include <net/ethernet.h>
#include <TargetConditionals.h>

typedef struct WiFiInfo *WiFiInfoRef;

typedef CF_ENUM(uint32_t, WiFiAuthType) {
	kWiFiAuthTypeNone 	= 0x0000,
	kWiFiAuthTypeUnknown	= 0xffff,
};

typedef CF_ENUM(uint8_t, WiFiInfoComparisonResult) {
	kWiFiInfoComparisonResultUnknown	= 0,
	kWiFiInfoComparisonResultSameNetwork	= 1,
	kWiFiInfoComparisonResultNetworkChanged	= 2,
	kWiFiInfoComparisonResultBSSIDChanged 	= 3,
};


const char *
WiFiAuthTypeGetString(WiFiAuthType auth_type);

const char *
WiFiInfoComparisonResultGetString(WiFiInfoComparisonResult result);

WiFiInfoRef
WiFiInfoCopy(CFStringRef ifname);

CFStringRef
WiFiInfoGetSSID(WiFiInfoRef w);

const struct ether_addr *
WiFiInfoGetBSSID(WiFiInfoRef w);

CFStringRef
WiFiInfoGetBSSIDString(WiFiInfoRef w);

WiFiAuthType
WiFiInfoGetAuthType(WiFiInfoRef w);

CFStringRef
WiFiInfoGetNetworkID(WiFiInfoRef w);

WiFiInfoComparisonResult
WiFiInfoCompare(WiFiInfoRef info1, WiFiInfoRef info2);

bool
WiFiInfoAllowSharingDeviceType(WiFiInfoRef info);

#if TARGET_OS_OSX
void
WiFiInfoSetHideBSSID(bool hide);
#endif /* TARGET_OS_OSX */

#endif /* _S_WIRELESS_H */
