/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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
#import "IPConfigurationParser.h"

#define TokenLinkStatus "linkStatus"
#define TokenSSID "ssid"
#define TokenMessage "message"
#define TokenAddress "address"
#define TokenComponentName "component"

@implementation IPConfigurationParser

- (instancetype)init
{
	NSArray<EFLogEventMatch *> *matches = nil;
	EFLogEventParser *parser = nil;

	matches = @[
		[[EFLogEventMatch alloc] initWithPattern:@"(?<"TokenInterfaceName">\\w+) link (?<"TokenLinkStatus">ACTIVE|INACTIVE)"
					 newEventHandler:
		 ^EFEvent *(NSTextCheckingResult *matchResult, EFLogEvent *logEvent, BOOL *isComplete) {
			 NSString *statusString = [logEvent substringForCaptureGroup:@TokenLinkStatus inMatchResult:matchResult];
			 EFNetworkControlPathEvent *newEvent = [self createInterfaceEventWithLogEvent:logEvent matchResult:matchResult];
			 if ([statusString isEqualToString:@"ACTIVE"]) {
				 newEvent.link = @"link up";
			 } else {
				 newEvent.link = @"link down";
			 }
			 *isComplete = YES;
			 return newEvent;
		 }],
		[[EFLogEventMatch alloc] initWithPattern:@"(?<"TokenInterfaceName">\\w+): SSID (?<"TokenSSID">\\S+) BSSID"
					 newEventHandler:
		 ^EFEvent *(NSTextCheckingResult *matchResult, EFLogEvent *logEvent, BOOL *isComplete) {
			 NSString *ssid = [logEvent substringForCaptureGroup:@TokenSSID inMatchResult:matchResult];
			 EFNetworkControlPathEvent *newEvent = [self createInterfaceEventWithLogEvent:logEvent matchResult:matchResult];
			 EFSubEvent *subEvent = [[EFSubEvent alloc] initWithTimestamp:logEvent.date textDescription:ssid];
			 [newEvent addSubEvent:subEvent];
			 *isComplete = YES;
			 return newEvent;
		 }],
		[[EFLogEventMatch alloc] initWithPattern:@"\\[(?<"TokenComponentName">\\w+ )?(?<"TokenInterfaceName">\\w+)\\] (?<"TokenMessage">(?:Transmit|Receive) \\d+ byte packet xid \\w+ (?:to|from) .*)"
					 newEventHandler:
		 ^EFEvent *(NSTextCheckingResult *matchResult, EFLogEvent *logEvent, BOOL *isComplete) {
			 NSString *message = [logEvent substringForCaptureGroup:@TokenMessage inMatchResult:matchResult];
			 NSString *component = [logEvent substringForCaptureGroup:@TokenComponentName inMatchResult:matchResult];
			 EFNetworkControlPathEvent *newEvent = [self createInterfaceEventWithLogEvent:logEvent matchResult:matchResult];
			 NSString *description = [[NSString alloc] initWithFormat:@"%@ %@", component, message];
			 EFSubEvent *subEvent = [[EFSubEvent alloc] initWithTimestamp:logEvent.date textDescription:description];
			 [newEvent addSubEvent:subEvent];
			 *isComplete = YES;
			 return newEvent;
		 }],
		[[EFLogEventMatch alloc] initWithPattern:@"\\w+ (?<"TokenInterfaceName">\\w+): setting (?<"TokenAddress">\\S+) netmask \\S+ broadcast \\S+"
					 newEventHandler:
		 ^EFEvent *(NSTextCheckingResult *matchResult, EFLogEvent *logEvent, BOOL *isComplete) {
			NSString *addressString = nil;

			*isComplete = YES;
			 addressString = [logEvent substringForCaptureGroup:@TokenAddress inMatchResult:matchResult];
			 if (addressString.length > 0) {
				 EFNetworkControlPathEvent *newEvent = [self createInterfaceEventWithLogEvent:logEvent matchResult:matchResult];
				 [self addAddress:addressString toInterfaceEvent:newEvent];
				 return newEvent;
			 }
			 return nil;
		 }],
		[[EFLogEventMatch alloc] initWithPattern:@"\\w+ (?<"TokenInterfaceName">\\w+): removing (?<"TokenAddress">.+)"
					 newEventHandler:
		 ^EFEvent *(NSTextCheckingResult *matchResult, EFLogEvent *logEvent, BOOL *isComplete) {
			NSString *addressString = nil;

			*isComplete = YES;
			 addressString = [logEvent substringForCaptureGroup:@TokenAddress inMatchResult:matchResult];
			 if (addressString.length > 0) {
				 EFNetworkControlPathEvent *newEvent = [self createInterfaceEventWithLogEvent:logEvent matchResult:matchResult];
				 if ([self removeAddress:addressString fromInterfaceEvent:newEvent]) {
					 return newEvent;
				 }
			 }
			 return nil;
		 }]
	];

	parser = [[EFLogEventParser alloc] initWithMatches:matches];
	return [super initWithCategory:@"IPConfiguration" eventParser:parser];
}

@end
