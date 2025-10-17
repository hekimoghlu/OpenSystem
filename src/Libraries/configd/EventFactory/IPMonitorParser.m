/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#import <Foundation/Foundation.h>
#import <EventFactory/EventFactory.h>

#import "IPMonitorParser.h"

#define TokenAddressAction "addressAction"
#define TokenAddress "address"
#define TokenServiceID "serviceID"
#define TokenV4Changes "v4Changes"
#define TokenV6Changes "v6Changes"

@interface IPMonitorParser ()
@property (readonly, nonatomic) NSRegularExpression *netChangeRegex;
@end

@implementation IPMonitorParser

- (instancetype)init
{
	NSError *regexError = nil;
	NSArray<EFLogEventMatch *> *matches = nil;
	EFLogEventParser *parser = nil;

	self = [super init];
	if (self == nil) {
		return nil;
	}

	_netChangeRegex = [[NSRegularExpression alloc] initWithPattern:@"(?<"TokenInterfaceName">\\w+)(?<"TokenAddressAction">[\\+\\-\\!\\/\\\\]?)(:(?<"TokenAddress">[:.0-9a-f]+))?" options:0 error:&regexError];
	if (_netChangeRegex == nil) {
		specs_log_err("Failed to create the network change regex: %@", regexError);
		return nil;
	}

	matches = @[
		[[EFLogEventMatch alloc] initWithPattern:@"\\d+. (?<"TokenInterfaceName">\\w+) serviceID=(?<"TokenServiceID">[-\\w]+) addr=(?<"TokenAddress">) rank=\\w+"
					 newEventHandler:
		 ^EFEvent *(NSTextCheckingResult *matchResult, EFLogEvent *logEvent, BOOL *isComplete) {
			EFNetworkControlPathEvent *newEvent = nil;
			NSString *serviceID = nil;
			NSString *addressString = nil;

			 *isComplete = YES;
			serviceID = [logEvent substringForCaptureGroup:@TokenServiceID inMatchResult:matchResult];
			addressString = [logEvent substringForCaptureGroup:@TokenAddress inMatchResult:matchResult];
			 if (serviceID != nil && addressString != nil) {
				 newEvent = [self createInterfaceEventWithLogEvent:logEvent matchResult:matchResult];
				 if (newEvent != nil) {
					 newEvent.serviceID = serviceID;
					 [self addAddress:addressString toInterfaceEvent:newEvent];
				 }
			 }
			 return newEvent;
		 }],
		[[EFLogEventMatch alloc] initWithPattern:@"network changed:( v4\\((?<"TokenV4Changes">[^\\)]+)\\))?( v6\\((?<"TokenV6Changes">[^\\)]+)\\))?"
				 multipleNewEventHandler:
		 ^NSArray<EFEvent *> *(NSTextCheckingResult *matchResult, EFLogEvent *logEvent) {
			 NSMutableDictionary<NSString *, EFNetworkControlPathEvent *> *newEventsMap = nil;
			 NSArray<NSString *> *tokens = @[ @TokenV4Changes, @TokenV6Changes ];
			 for (NSString *token in tokens) {
				 NSArray<NSTextCheckingResult *> *matches = nil;
				 BOOL isPrimary = YES;
				 BOOL isIPv4 = [token isEqualToString:@TokenV4Changes];
				 NSString *changes = [logEvent substringForCaptureGroup:token inMatchResult:matchResult];
				 if (changes == nil) {
					 continue;
				 }
				 matches = [self.netChangeRegex matchesInString:changes options:0 range:NSMakeRange(0, changes.length)];
				 for (NSTextCheckingResult *match in matches) {
					 EFNetworkControlPathEvent *event = nil;

					 NSString *interfaceName = [self substringOfString:changes forCaptureGroup:@TokenInterfaceName inMatchResult:match];
					 NSString *addressAction = [self substringOfString:changes forCaptureGroup:@TokenAddressAction inMatchResult:match];
					 NSString *address = [self substringOfString:changes forCaptureGroup:@TokenAddress inMatchResult:match];

					 if (interfaceName == nil || address.length == 0) {
						 continue;
					 }

					 event = newEventsMap[interfaceName];
					 if (event == nil) {
						 event = [self createInterfaceEventWithLogEvent:logEvent interfaceName:interfaceName];
						 if (newEventsMap == nil) {
							 newEventsMap = [[NSMutableDictionary alloc] init];
						 }
						 newEventsMap[interfaceName] = event;
					 }

					 if (addressAction.length > 0 && [addressAction isEqualToString:@"-"]) {
						 if (isPrimary) {
							 if (isIPv4) {
								 event.primaryStateIPv4 = EFPrimaryStateNotPrimary;
							 } else {
								 event.primaryStateIPv6 = EFPrimaryStateNotPrimary;
							 }
						 }
						 [self removeAddress:address fromInterfaceEvent:event];
					 } else {
						 if (isPrimary) {
							 if (isIPv4) {
								 event.primaryStateIPv4 = EFPrimaryStatePrimary;
							 } else {
								 event.primaryStateIPv6 = EFPrimaryStatePrimary;
							 }
							 for (NSString *otherInterfaceName in SCLogParser.interfaceMap) {
								 EFNetworkControlPathEvent *otherEvent = nil;

								 if ([otherInterfaceName isEqualToString:interfaceName]) {
									 continue;
								 }
								 otherEvent = newEventsMap[otherInterfaceName];
								 if (otherEvent == nil) {
									 otherEvent = [self createInterfaceEventWithLogEvent:logEvent interfaceName:otherInterfaceName];
									 if (newEventsMap == nil) {
										 newEventsMap = [[NSMutableDictionary alloc] init];
									 }
									 newEventsMap[otherInterfaceName] = otherEvent;
								 }
								 if (isIPv4) {
									 otherEvent.primaryStateIPv4 = EFPrimaryStateNotPrimary;
								 } else {
									 otherEvent.primaryStateIPv6 = EFPrimaryStateNotPrimary;
								 }
							 }
							 isPrimary = NO;
						 }
						 [self addAddress:address toInterfaceEvent:event];
					 }
				 }
			 }
			 return newEventsMap.allValues;
		 }],
	];
	parser = [[EFLogEventParser alloc] initWithMatches:matches];
	return [super initWithCategory:@"IPMonitor" eventParser:parser];
}

@end
