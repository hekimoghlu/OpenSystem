/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
#include <sys/types.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <SystemConfiguration/CaptiveNetwork.h>
#include <SystemConfiguration/SCPrivate.h>

#pragma mark -
#pragma mark CaptiveNetwork.framework APIs (exported through the SystemConfiguration.framework)

const CFStringRef kCNNetworkInfoKeySSIDData    = CFSTR("SSIDDATA");
const CFStringRef kCNNetworkInfoKeySSID        = CFSTR("SSID");
const CFStringRef kCNNetworkInfoKeyBSSID       = CFSTR("BSSID");

static void *
__loadCaptiveNetwork(void) {
	static void		*image	= NULL;
	static dispatch_once_t	once;

	dispatch_once(&once, ^{
		image = _SC_dlopen("/System/Library/PrivateFrameworks/CaptiveNetwork.framework/CaptiveNetwork");
	});

	return image;
}

Boolean
CNSetSupportedSSIDs(CFArrayRef ssidArray)
{
	static typeof (CNSetSupportedSSIDs) *dyfunc = NULL;
	if (!dyfunc) {
		void *image = __loadCaptiveNetwork();
		if (image) dyfunc = (typeof (CNSetSupportedSSIDs) *)dlsym(image, "__CNSetSupportedSSIDs");
	}
	return dyfunc ? dyfunc(ssidArray) : FALSE;
}

Boolean
CNMarkPortalOnline(CFStringRef interfaceName)
{
	static typeof (CNMarkPortalOnline) *dyfunc = NULL;
	if (!dyfunc) {
		void *image = __loadCaptiveNetwork();
		if (image) dyfunc = (typeof (CNMarkPortalOnline) *)dlsym(image, "__CNMarkPortalOnline");
	}
	return dyfunc ? dyfunc(interfaceName) : FALSE;
}

Boolean
CNMarkPortalOffline(CFStringRef interfaceName)
{
	static typeof (CNMarkPortalOffline) *dyfunc = NULL;
	if (!dyfunc) {
		void *image = __loadCaptiveNetwork();
		if (image) dyfunc = (typeof (CNMarkPortalOffline) *)dlsym(image, "__CNMarkPortalOffline");
	}
	return dyfunc ? dyfunc(interfaceName) : FALSE;
}

CFArrayRef
CNCopySupportedInterfaces(void)
{
	static typeof (CNCopySupportedInterfaces) *dyfunc = NULL;
	if (!dyfunc) {
		void *image = __loadCaptiveNetwork();
		if (image) dyfunc = (typeof (CNCopySupportedInterfaces) *)dlsym(image, "__CNCopySupportedInterfaces");
	}
	return dyfunc ? dyfunc() : NULL;
}

CFDictionaryRef
CNCopyCurrentNetworkInfo(CFStringRef interfaceName)
{
	static typeof (CNCopyCurrentNetworkInfo) *dyfunc = NULL;
	if (!dyfunc) {
		void *image = __loadCaptiveNetwork();
		if (image) dyfunc = (typeof (CNCopyCurrentNetworkInfo) *)dlsym(image, "__CNCopyCurrentNetworkInfo");
	}
	return dyfunc ? dyfunc(interfaceName) : NULL;
}

