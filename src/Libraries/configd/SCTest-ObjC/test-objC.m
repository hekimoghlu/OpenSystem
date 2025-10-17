/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
 * Modification History
 *
 * April 21, 2015		Sushant Chavan
 * - initial revision
 */

/*
 *  A Objective-C test target to test SC APIs
 */

#import <TargetConditionals.h>

#if	!defined(USING_PUBLIC_SDK)
@import Foundation;
@import SystemConfiguration;
@import SystemConfiguration_Private;
#else	// !defined(USING_PUBLIC_SDK)
#include <Foundation/Foundation.h>
#include <SystemConfiguration/SystemConfiguration.h>
#endif	// !defined(USING_PUBLIC_SDK)

#if	TARGET_OS_MACCATALYST
#pragma message "Building for IOS_MAC"
#endif

#define MY_APP_NAME	CFSTR("SCTestObjC")
#define TARGET_HOST	"www.apple.com"


#if	!TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)
static void
test_SCDynamicStore(void)
{
	CFDictionaryRef		dict;
	CFStringRef		intf;
	CFStringRef		key;
	SCDynamicStoreRef	store;
	NSLog(@"\n\n*** SCDynamicStore ***\n\n");

	store = SCDynamicStoreCreate(NULL, MY_APP_NAME, NULL, NULL);
	key = SCDynamicStoreKeyCreateNetworkGlobalEntity(NULL, kSCDynamicStoreDomainState, kSCEntNetIPv4);
	dict = SCDynamicStoreCopyValue(store, key);
	intf = CFDictionaryGetValue(dict, kSCDynamicStorePropNetPrimaryInterface);
	NSLog(@"- Primary Interface is %@\n", intf);

	CFRelease(store);
	CFRelease(dict);
	CFRelease(key);
}
#endif	// !TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)

#if	!TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)
static void
test_SCNetworkConfiguration(void)
{
	CFIndex			count;
	CFIndex			idx;
	CFArrayRef		interfaces;
	NSLog(@"\n\n*** SCNetworkConfiguration ***\n\n");
	
	interfaces = SCNetworkInterfaceCopyAll();
	count = CFArrayGetCount(interfaces);
	NSLog(@"Network Interfaces:\n");
	for (idx=0; idx < count; idx++) {
		SCNetworkInterfaceRef intf;
		CFStringRef bsdName;
		
		intf = CFArrayGetValueAtIndex(interfaces, idx);
		bsdName = SCNetworkInterfaceGetBSDName(intf);
		NSLog(@"- %@", bsdName);
	}
	
	CFRelease(interfaces);
}
#endif	// !TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)

static void
test_SCNetworkReachability(void)
{
	SCNetworkReachabilityFlags	flags;
	SCNetworkReachabilityRef	target;
	NSLog(@"\n\n*** SCNetworkReachability ***\n\n");

	target = SCNetworkReachabilityCreateWithName(NULL, TARGET_HOST);
	(void)SCNetworkReachabilityGetFlags(target, &flags);
	NSLog(@"- Reachability flags for "TARGET_HOST": %#x", flags);

	CFRelease(target);
}

#if	!TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)
static void
test_SCPreferences(void)
{
	CFIndex			count;
	CFIndex			idx;
	CFStringRef		model = NULL;
	SCPreferencesRef	prefs;
	CFArrayRef		services;
	NSLog(@"\n\n*** SCPreferences ***\n\n");
	
	prefs = SCPreferencesCreate(NULL, MY_APP_NAME, NULL);
	model = SCPreferencesGetValue(prefs, CFSTR("Model"));
	if (model != NULL) {
		NSLog(@"Current model is %@", model);
	}
	
	services = SCNetworkServiceCopyAll(prefs);
	count = CFArrayGetCount(services);
	NSLog(@"Network Services:\n");
	for (idx = 0; idx < count; idx++) {
		SCNetworkServiceRef serv;
		CFStringRef servName;
		
		serv = CFArrayGetValueAtIndex(services, idx);
		servName = SCNetworkServiceGetName(serv);
		NSLog(@"- %@\n", servName);
	}
	
	CFRelease(prefs);
	CFRelease(services);
}
#endif	// !TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)

static void
SCTest(void)
{

#if	!TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)
	test_SCDynamicStore();
#endif	// !TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)

#if	!TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)
	test_SCNetworkConfiguration();
#endif	// !TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)

	test_SCNetworkReachability();

#if	!TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)
	test_SCPreferences();
#endif	// !TARGET_OS_SIMULATOR && !defined(USING_PUBLIC_SDK)

}

int
main(int argc, const char * argv[]) {
#pragma unused(argc, argv)

#if	TARGET_OS_MACCATALYST
#if	!defined(USING_PUBLIC_SDK)
#include <CoreFoundation/CFPriv.h>
#else	// !defined(USING_PUBLIC_SDK)
extern Boolean _CFMZEnabled(void);
#endif	// !defined(USING_PUBLIC_SDK)
	if (_CFMZEnabled()) {
		NSLog(@"*** IOS_MAC ***\n");
	}
#endif

	@autoreleasepool {
		SCTest();
	}
	return 0;
}
