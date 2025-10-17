/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#import <XCTest/XCTest.h>

#import "test_base.h"

@interface IPv6AutomaticConfigTest : XCTestCase
@property (nonatomic) IPConfigurationFrameworkTestBase * base;
@end

@implementation IPv6AutomaticConfigTest

- (void)setUp 
{
	[self setBase:[IPConfigurationFrameworkTestBase sharedInstance]];
}

- (void)tearDown 
{
	// pass
}

#pragma mark -
#pragma mark IPv6 Auto (non-DHCPv6)

- (void)testIPv6MethodAutomatic
{
	long err = 0;
	bool setKeys = false;
	IPConfigurationServiceRef service = NULL;
	NSDictionary *customServiceProperties = @{
		BRIDGED_NSSTRINGREF(kIPConfigurationServiceOptionIPv6Entity) : @{
			BRIDGED_NSSTRINGREF(kSCPropNetIPv6ConfigMethod) : BRIDGED_NSSTRINGREF(kSCValNetIPv6ConfigMethodAutomatic)
		}
	};

	XCT_REQUIRE_NONNULL(self.base.ifname, done, "Failed to get available interface");
	NSLogDebug(@"Custom properties for new IPConfigurationService on '%@':\n%@", self.base.ifname, customServiceProperties);
	service = IPConfigurationServiceCreate((__bridge CFStringRef)self.base.ifname, (__bridge CFDictionaryRef)customServiceProperties);
	XCTAssertNotNil(BRIDGED_ID(service), "Failed to create AUTOMATIC-V6 service");
	NSLogDebug(@"Successfully instantiated IPConfigurationService with key '%@'", IPConfigurationServiceGetNotificationKey(service));
	[[self base] setService:service];
	[[self base] setAlternativeValidation:TRUE];
	setKeys = SCDynamicStoreSetNotificationKeys([[self base] store], NULL, SCD_KEY_PATTERNS_CFARRAYREF);
	XCT_REQUIRE_TRUE(setKeys, done, "Failed to set SCDynamicStore notification keys");
	err = dispatch_semaphore_wait([[self base] serviceSem], dispatch_time(DISPATCH_TIME_NOW, SEMAPHORE_TIMEOUT_NANOSEC));
	XCTAssertEqual(err, 0, "Semaphore timed out waiting for service to be up");

done:
	[[self base] setService:NULL];
	RELEASE_NULLSAFE(service);
}

#pragma mark -
#pragma mark IPv6 Link-Local


- (void)testIPv6MethodLinkLocalWithAdditionalOptions
{
	long err = 0;
	bool setKeys = false;
	IPConfigurationServiceRef service = NULL;
	NSDictionary *servicePropertiesFromStore = nil;
	NSDictionary *customServiceProperties = @{
		BRIDGED_NSSTRINGREF(kIPConfigurationServiceOptionPerformNUD) : [NSNumber numberWithBool:TRUE],
		BRIDGED_NSSTRINGREF(kIPConfigurationServiceOptionEnableDAD) : [NSNumber numberWithBool:TRUE],
		BRIDGED_NSSTRINGREF(kIPConfigurationServiceOptionEnableCLAT46) : [NSNumber numberWithBool:TRUE],
		BRIDGED_NSSTRINGREF(kIPConfigurationServiceOptionIPv6Entity) : @{
			BRIDGED_NSSTRINGREF(kSCPropNetIPv6ConfigMethod) : BRIDGED_NSSTRINGREF(kSCValNetIPv6ConfigMethodLinkLocal)
		}
	};
#define IPV6_LINKLOCAL_ADDR_PREFIX "fe80"
#define IPV4_LINKLOCAL_PREFIXLEN 64

	XCT_REQUIRE_NONNULL(self.base.ifname, done, "Failed to get available interface");
	NSLogDebug(@"Custom properties for new IPConfigurationService on '%@':\n%@", self.base.ifname, customServiceProperties);
	service = IPConfigurationServiceCreate((__bridge CFStringRef)self.base.ifname, (__bridge CFDictionaryRef)customServiceProperties);
	XCT_REQUIRE_NONNULL(BRIDGED_ID(service), done, "Failed to create manual IPv6 service with unique local addr");
	[[self base] setService:service];
	NSLogDebug(@"Successfully instantiated IPConfigurationService with key '%@'", IPConfigurationServiceGetNotificationKey(service));
	setKeys = SCDynamicStoreSetNotificationKeys([[self base] store], SCD_KEY_IPCONFIGSERVICEKEY_CFARRAYREF(service), SCD_KEY_PATTERNS_CFARRAYREF);
	XCT_REQUIRE_TRUE(setKeys, done, "Failed to set SCDynamicStore notification keys");
	err = dispatch_semaphore_wait([[self base] serviceSem], dispatch_time(DISPATCH_TIME_NOW, SEMAPHORE_TIMEOUT_NANOSEC));
	XCT_REQUIRE_NERROR(err, done, "Semaphore timed out waiting for service to be up");
	NSLogDebug(@"Waited semaphore, retrying getting service properties");
	servicePropertiesFromStore = (__bridge_transfer NSDictionary *)IPConfigurationServiceCopyInformation(service);

	// validates that the configuration set conforms to ipv6 link-local
	XCT_REQUIRE_NONNULL(servicePropertiesFromStore, done, "Failed to retrieve active IPConfigurationService properties");
	XCT_REQUIRE_NONNULL(BRIDGED_ID(isA_CFDictionary((__bridge CFDictionaryRef)servicePropertiesFromStore)),
			    done, "Failed to get a CFDictionary type from the SCDynamicStore");
	NSLogDebug(@"New IPConfigurationService properties from the store:\n%@", servicePropertiesFromStore);
	XCTAssertEqualObjects(self.base.ifname,
			      [[servicePropertiesFromStore objectForKey:BRIDGED_NSSTRINGREF(kSCEntNetIPv6)] objectForKey:BRIDGED_NSSTRINGREF(kSCPropInterfaceName)],
			      "Failed to assert network interface name");
	XCTAssertTrue([[[[servicePropertiesFromStore
			  objectForKey:BRIDGED_NSSTRINGREF(kSCEntNetIPv6)]
			 objectForKey:BRIDGED_NSSTRINGREF(kSCPropNetIPv6Addresses)]
			objectAtIndex:0]
		       containsString:@IPV6_LINKLOCAL_ADDR_PREFIX],
		      "Failed to assert link-local address");
	XCTAssertEqualObjects(@[@IPV4_LINKLOCAL_PREFIXLEN],
			      [[servicePropertiesFromStore
				objectForKey:BRIDGED_NSSTRINGREF(kSCEntNetIPv6)]
			       objectForKey:BRIDGED_NSSTRINGREF(kSCPropNetIPv6PrefixLength)],
			      "Failed to assert link-local prefix length");

done:
	[[self base] setService:NULL];
	RELEASE_NULLSAFE(service);
}

@end
