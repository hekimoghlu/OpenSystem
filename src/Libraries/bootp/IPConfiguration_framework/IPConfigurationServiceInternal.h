/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 13, 2021.
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
 * IPConfigurationServiceInternal.h
 * - internal definitions
 */

/* 
 * Modification History
 *
 * April 14, 2011 	Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

#ifndef _IPCONFIGURATIONSERVICEINTERNAL_H
#define _IPCONFIGURATIONSERVICEINTERNAL_H

#include <CoreFoundation/CFString.h>
#include "symbol_scope.h"
#include "IPConfigurationService.h"
#include "ObjectWrapper.h"

#define kIPConfigurationServiceOptions	CFSTR("__IPConfigurationServiceOptions") /* dictionary */

#define _kIPConfigurationServiceOptionClearState	\
    CFSTR("ClearState")		/* boolean */
#define _kIPConfigurationServiceOptionEnableDAD	\
    CFSTR("EnableDAD")		/* boolean */
#define	_kIPConfigurationServiceOptionEnableCLAT46 \
    CFSTR("EnableCLAT46")	/* boolean */
#define	_kIPConfigurationServiceOptionEnableDHCPv6 \
    CFSTR("EnableDHCPv6")	/* boolean */
#define _kIPConfigurationServiceOptionMonitorPID	\
    CFSTR("MonitorPID") 	/* boolean */
#define _kIPConfigurationServiceOptionMTU	\
    CFSTR("MTU")		/* number */
#define _kIPConfigurationServiceOptionNoPublish	\
    CFSTR("NoPublish")		/* boolean */
#define _kIPConfigurationServiceOptionPerformNUD	\
    CFSTR("PerformNUD")		/* boolean */
#define _kIPConfigurationServiceOptionServiceID		\
    CFSTR("ServiceID")		/* string (UUID) */
#define _kIPConfigurationServiceOptionAPNName	\
    CFSTR("APNName")		/* string */

#define _kIPConfigurationServiceOptionIPv6Entity	\
    CFSTR("IPv6Entity")	/* dictionary */
#define _kIPConfigurationServiceOptionIPv4Entity	\
    CFSTR("IPv4Entity")	/* dictionary */

#define IPCONFIGURATION_SERVICE_FORMAT CFSTR("Plugin:IPConfigurationService:%@")

INLINE CFStringRef
IPConfigurationServiceKey(CFStringRef serviceID)
{
    return (CFStringCreateWithFormat(NULL, NULL,
				     IPCONFIGURATION_SERVICE_FORMAT,
				     serviceID));
}

IPConfigurationServiceRef
IPConfigurationServiceCreateInternal(CFStringRef interface_name,
				     CFDictionaryRef options);
Boolean
IPConfigurationServiceIsValid(IPConfigurationServiceRef service);

Boolean
IPConfigurationServiceStart(IPConfigurationServiceRef service);

SCDynamicStoreRef
store_create(const void * object,
	     CFStringRef label,
	     dispatch_queue_t queue,
	     SCDynamicStoreCallBack change_callback,
	     SCDynamicStoreDisconnectCallBack disconnect_callback,
	     ObjectWrapperRef * ret_wrapper);
#endif /* _IPCONFIGURATIONSERVICEINTERNAL_H */
