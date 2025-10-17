/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
#ifndef _IPMONITOR_CONTROL_H
#define _IPMONITOR_CONTROL_H
/*
 * IPMonitorControl.h
 * - IPC channel to IPMonitor
 * - used to create interface rank assertions
 */

/*
 * Modification History
 *
 * December 16, 2013	Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

#include <SystemConfiguration/SCNetworkConfigurationPrivate.h>

struct IPMonitorControl;
typedef struct IPMonitorControl * IPMonitorControlRef;

IPMonitorControlRef
IPMonitorControlCreate(void);

/**
 ** Interface Rank Assertion
 **/

Boolean
IPMonitorControlSetInterfacePrimaryRank(IPMonitorControlRef control,
					CFStringRef ifname,
					SCNetworkServicePrimaryRank rank);

SCNetworkServicePrimaryRank
IPMonitorControlGetInterfacePrimaryRank(IPMonitorControlRef control,
					CFStringRef ifname);

CFStringRef
IPMonitorControlCopyInterfaceRankAssertionNotificationKey(CFStringRef ifname);

typedef CFTypeRef InterfaceRankAssertionInfoRef;

SCNetworkServicePrimaryRank
InterfaceRankAssertionInfoGetPrimaryRank(InterfaceRankAssertionInfoRef info);

pid_t
InterfaceRankAssertionInfoGetProcessID(InterfaceRankAssertionInfoRef info);

CFStringRef
InterfaceRankAssertionInfoGetProcessName(InterfaceRankAssertionInfoRef info);

CFArrayRef /* of InterfaceRankAssertionInfoRef */
IPMonitorControlCopyInterfaceRankAssertionInfo(IPMonitorControlRef control,
					       CFStringRef ifname);
CFArrayRef /* of CFString */
IPMonitorControlCopyInterfaceRankAssertionInterfaceNames(IPMonitorControlRef control);

/**
 ** Interface Advisory
 **/
Boolean
IPMonitorControlSetInterfaceAdvisory(IPMonitorControlRef control,
				     CFStringRef ifname,
				     SCNetworkInterfaceAdvisory advisory,
				     CFStringRef reason);

Boolean
IPMonitorControlIsInterfaceAdvisorySet(IPMonitorControlRef control,
				       CFStringRef ifname,
				       SCNetworkInterfaceAdvisory advisory);

CFStringRef
IPMonitorControlCopyInterfaceAdvisoryNotificationKey(CFStringRef ifname);

Boolean
IPMonitorControlAnyInterfaceAdvisoryIsSet(IPMonitorControlRef control);

typedef CFTypeRef InterfaceAdvisoryInfoRef;

SCNetworkInterfaceAdvisory
InterfaceAdvisoryInfoGetAdvisory(InterfaceAdvisoryInfoRef info);

pid_t
InterfaceAdvisoryInfoGetProcessID(InterfaceAdvisoryInfoRef info);

CFStringRef
InterfaceAdvisoryInfoGetProcessName(InterfaceAdvisoryInfoRef info);

CFArrayRef /* of InterfaceAdvisoryInfoRef */
IPMonitorControlCopyInterfaceAdvisoryInfo(IPMonitorControlRef control,
					  CFStringRef ifname);

CFArrayRef /* of CFString */
IPMonitorControlCopyInterfaceAdvisoryInterfaceNames(IPMonitorControlRef control);

#endif /* _IPMONITOR_CONTROL_H */
