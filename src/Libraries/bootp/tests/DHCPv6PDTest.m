/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
#import "test_dhcpv6_icmpv6.h"

@interface DHCPv6PDTest : XCTestCase
@property (nonatomic, weak) IPConfigurationFrameworkTestBase * base;
@property (nonatomic, strong) DHCPv6Server * dhcpv6Server;
@property NSString * serviceInfoString;
@end

@implementation DHCPv6PDTest

- (void)setUp
{
	NSString *dhcpServerIfname = nil;
	NSString *dhcpClientIfname = nil;

	self.base = [IPConfigurationFrameworkTestBase sharedInstance];
	setupForMockDHCPExchange(&dhcpServerIfname, &dhcpClientIfname);
	if (dhcpServerIfname != nil && dhcpClientIfname != nil) {
		self.base.dhcpServerIfname = dhcpServerIfname;
		self.base.dhcpClientIfname = dhcpClientIfname;
	}
}

- (void)tearDown
{
	cleanupFromMockDHCPExchange(self.base.dhcpServerIfname, self.base.dhcpClientIfname);
}

#pragma mark -
#pragma mark DHCPv6PD

static DHCPv6PDServiceHandler myDHCPv6PDServiceHandler(id myself)
{
	__weak DHCPv6PDTest *instanceRef = (DHCPv6PDTest *)myself;
	
	return ^void(Boolean valid, DHCPv6PDServiceInfoRef service_info, CFDictionaryRef info) {
		NSLogDebug(@"%s: %s", __func__, DHCPV6PD_SERVICE_QUEUE);
		if (valid && service_info != NULL) {
			NSString *serviceInfoStr = nil;
			struct in6_addr prefixAddr = { 0 };
			uint8_t prefixLen = 0;

			if (instanceRef == nil) {
				NSLog(@"%s: invalid test object", __func__);
				return;
			}
			DHCPv6PDServiceInfoGetPrefix(service_info, &prefixAddr);
			prefixLen = DHCPv6PDServiceInfoGetPrefixLength(service_info);;
			serviceInfoStr = pdServiceStringFromPrefixAndLength(&prefixAddr, prefixLen);
			NSLogDebug(@"serviceInfoString: %@", serviceInfoStr);
			[myself setServiceInfoString:serviceInfoStr];
			dispatch_semaphore_signal([[instanceRef base] serviceSem]);
		} else if (valid && service_info == NULL) {
			NSLog(@"%s: link status down or unresponsive DHCPv6 server", __func__);
		} else if (!valid && service_info == NULL) {
			NSLog(@"%s: PD service invalid", __func__);
		} else {
			NSLog(@"%s: undefined status", __func__);
		}
	};
}

#define TEST_DHCPV6PD_PREFIXLEN 96
#define TEST_DHCPV6PD_PREFIX_IN6ADDR \
{{{ 0xfc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02 }}}

- (void)testConfigureFethInterfaceWithDHCPv6PDService
{
	long err = 0;
	DHCPv6PDServiceRef service = NULL;
	dispatch_queue_t queue = NULL;
	DHCPv6PDServiceHandler handler = NULL;
	const struct in6_addr prefix = TEST_DHCPV6PD_PREFIX_IN6ADDR;
	/* fc00:0:0:0:0:2::/96 */
	NSString *expectedPDServiceStr = pdServiceStringFromPrefixAndLength(&prefix, TEST_DHCPV6PD_PREFIXLEN);

	[self setDhcpv6Server:[[DHCPv6Server alloc] initWithFailureMode:kDHCPServerFailureModeNone andInterface:self.base.dhcpServerIfname]];
	service = DHCPv6PDServiceCreate((__bridge CFStringRef)self.base.dhcpClientIfname, &prefix, TEST_DHCPV6PD_PREFIXLEN, NULL);
	XCT_REQUIRE_NONNULL(BRIDGED_ID(service), done, "Failed to create DHCPv6 service");
	[[self base] setPdServiceKey:expectedPDServiceStr];
	queue = dispatch_queue_create(DHCPV6PD_SERVICE_QUEUE, NULL);
	XCT_REQUIRE_NONNULL(queue, done, "Failed to create dispatch queue for service");
	handler = myDHCPv6PDServiceHandler(self);
	DHCPv6PDServiceSetQueueAndHandler(service, queue, handler);
	DHCPv6PDServiceResume(service);
	err = dispatch_semaphore_wait([[self base] serviceSem], dispatch_time(DISPATCH_TIME_NOW, SEMAPHORE_TIMEOUT_NANOSEC));
	XCT_REQUIRE_NERROR(err, done, "Failed to get signal that PD service is up, semaphore timed out");

	// validates that the active PD service is exactly the one we requested
	// note that the DHCPv6PDServiceHandler sets the self.serviceInfoString
	XCTAssertEqualObjects(expectedPDServiceStr, [self serviceInfoString], "Failed to validate prefix delegation service");

done:
	[[self dhcpv6Server] disconnect];
	[self setDhcpv6Server:nil];
	[[self base] setPdServiceKey:nil];
	RELEASE_NULLSAFE(service);
}


- (void)testConfigureFethInterfaceWithDHCPv6PDServiceFailsWithNotOnLinkAndExponentialBackoff
{
	long err = 0;
	DHCPv6PDServiceRef service = NULL;
	dispatch_queue_t queue = NULL;
	DHCPv6PDServiceHandler handler = NULL;
	/* fc00:0:0:0:0:2::/96 */
	const struct in6_addr prefix = TEST_DHCPV6PD_PREFIX_IN6ADDR;
	NSString *expectedPDServiceStr = pdServiceStringFromPrefixAndLength(&prefix, TEST_DHCPV6PD_PREFIXLEN);

	[self setDhcpv6Server:[[DHCPv6Server alloc] initWithFailureMode:kDHCPServerFailureModeNotOnLink andInterface:self.base.dhcpServerIfname]];
	service = DHCPv6PDServiceCreate((__bridge CFStringRef)self.base.dhcpClientIfname, &prefix, TEST_DHCPV6PD_PREFIXLEN, NULL);
	XCT_REQUIRE_NONNULL(BRIDGED_ID(service), done, "Failed to create DHCPv6 service");
	[[self base] setPdServiceKey:expectedPDServiceStr];
	queue = dispatch_queue_create(DHCPV6PD_SERVICE_QUEUE, NULL);
	XCT_REQUIRE_NONNULL(queue, done, "Failed to create dispatch queue for service");
	handler = myDHCPv6PDServiceHandler(self);
	DHCPv6PDServiceSetQueueAndHandler(service, queue, handler);
	DHCPv6PDServiceResume(service);

	// validates that the client decreased its subsequent requests frequency exponentially (exponential backoff)
	err = dispatch_semaphore_wait([[self dhcpv6Server] exponentialBackoffSem],
				      dispatch_time(DISPATCH_TIME_NOW, SEMAPHORE_TIMEOUT_NANOSEC));
	XCTAssertTrue(err == 0, "Failed to get signal that the client performed exponential backoff");

	// validates that the semaphore timed out because the client never got a successful response from the server
	err = dispatch_semaphore_wait([[self base] serviceSem], 
				      dispatch_time(DISPATCH_TIME_NOW, SEMAPHORE_TIMEOUT_NANOSEC_SHORT));
	XCTAssertTrue(err != 0, "Failed to time out PD service semaphore");

done:
	[[self dhcpv6Server] disconnect];
	[self setDhcpv6Server:nil];
	[[self base] setPdServiceKey:nil];
	RELEASE_NULLSAFE(service);
}

- (void)testConfigureFethInterfaceWithDHCPv6PDServiceFailsWithNoPrefixAvailAndExponentialBackoff
{
	long err = 0;
	DHCPv6PDServiceRef service = NULL;
	dispatch_queue_t queue = NULL;
	DHCPv6PDServiceHandler handler = NULL;
	/* fc00:0:0:0:0:2::/96 */
	const struct in6_addr prefix = TEST_DHCPV6PD_PREFIX_IN6ADDR;
	NSString *expectedPDServiceStr = pdServiceStringFromPrefixAndLength(&prefix, TEST_DHCPV6PD_PREFIXLEN);

	[self setDhcpv6Server:[[DHCPv6Server alloc] initWithFailureMode:kDHCPServerFailureModeNoPrefixAvail andInterface:self.base.dhcpServerIfname]];
	service = DHCPv6PDServiceCreate((__bridge CFStringRef)self.base.dhcpClientIfname, &prefix, TEST_DHCPV6PD_PREFIXLEN, NULL);
	XCT_REQUIRE_NONNULL(BRIDGED_ID(service), done, "Failed to create DHCPv6 service");
	[[self base] setPdServiceKey:expectedPDServiceStr];
	queue = dispatch_queue_create(DHCPV6PD_SERVICE_QUEUE, NULL);
	XCT_REQUIRE_NONNULL(queue, done, "Failed to create dispatch queue for service");
	handler = myDHCPv6PDServiceHandler(self);
	DHCPv6PDServiceSetQueueAndHandler(service, queue, handler);
	DHCPv6PDServiceResume(service);
	
	// validates that the client decreased its subsequent requests frequency exponentially (exponential backoff)
	err = dispatch_semaphore_wait([[self dhcpv6Server] exponentialBackoffSem],
				      dispatch_time(DISPATCH_TIME_NOW, SEMAPHORE_TIMEOUT_NANOSEC));
	XCTAssertTrue(err == 0, "Failed to get signal that the client performed exponential backoff");
	
	// validates that the semaphore timed out because the client never got a successful response from the server
	err = dispatch_semaphore_wait([[self base] serviceSem],
				      dispatch_time(DISPATCH_TIME_NOW, SEMAPHORE_TIMEOUT_NANOSEC_SHORT));
	XCTAssertTrue(err != 0, "Failed to time out PD service semaphore");
	
done:
	[[self dhcpv6Server] disconnect];
	[self setDhcpv6Server:nil];
	[[self base] setPdServiceKey:nil];
	RELEASE_NULLSAFE(service);
}

@end
