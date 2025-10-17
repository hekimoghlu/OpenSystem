/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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

#import "KernelEventMonitorParser.h"

#define TokenStatus "status"
#define TokenLinkStatus "linkStatus"
#define TokenLinkQuality "linkQuality"

@implementation KernelEventMonitorParser

- (instancetype)init
{
	NSArray<EFLogEventMatch *> *matches = @[
		[[EFLogEventMatch alloc] initWithPattern:@"Process interface (?<"TokenStatus">attach|detach): (?<"TokenInterfaceName">\\w+)"
					 newEventHandler:
		 ^EFEvent *(NSTextCheckingResult *matchResult, EFLogEvent *logEvent, BOOL *isComplete) {
			EFNetworkControlPathEvent *newEvent = nil;
			NSString *statusString = nil;

			 *isComplete = YES;
			 statusString = [logEvent substringForCaptureGroup:@TokenStatus inMatchResult:matchResult];
			 if (statusString != nil) {
				 newEvent = [self createInterfaceEventWithLogEvent:logEvent matchResult:matchResult];
				 if (newEvent != nil) {
					newEvent.interfaceStatus = statusString;
				 }
			 }
			 return newEvent;
		 }],
		[[EFLogEventMatch alloc] initWithPattern:@"Process interface link (?<"TokenLinkStatus">down|up): (?<"TokenInterfaceName">\\w+)"
					 newEventHandler:
		 ^EFEvent *(NSTextCheckingResult *matchResult, EFLogEvent *logEvent, BOOL *isComplete) {
			EFNetworkControlPathEvent *newEvent = nil;
			NSString *linkStatusString = nil;

			 *isComplete = YES;
			 linkStatusString = [logEvent substringForCaptureGroup:@TokenLinkStatus inMatchResult:matchResult];
			 if (linkStatusString != nil) {
				 newEvent = [self createInterfaceEventWithLogEvent:logEvent matchResult:matchResult];
				 if (newEvent != nil) {
					 if ([linkStatusString isEqualToString:@"up"]) {
						 newEvent.link = @"link up";
					 } else if ([linkStatusString isEqualToString:@"down"]) {
						 newEvent.link = @"link down";
					 } else {
						 newEvent.link = linkStatusString;
					 }
				 }
			 }
			 return newEvent;
		 }],
		[[EFLogEventMatch alloc] initWithPattern:@"Process interface quality: (?<"TokenInterfaceName">\\w+) \\(q=(?<"TokenLinkQuality">[-\\d]+)\\)"
					 newEventHandler:
		 ^EFEvent * _Nullable(NSTextCheckingResult * _Nonnull matchResult, EFLogEvent * _Nonnull logEvent, BOOL * _Nonnull isComplete) {
			EFNetworkControlPathEvent *newEvent = nil;
			NSString *qualityString = nil;

			 *isComplete = YES;
			 qualityString = [logEvent substringForCaptureGroup:@TokenLinkQuality inMatchResult:matchResult];
			 if (qualityString != nil) {
				 newEvent = [self createInterfaceEventWithLogEvent:logEvent matchResult:matchResult];
				 if (newEvent != nil) {
					newEvent.linkQuality = qualityString.integerValue;
				 }
			 }
			 return newEvent;
		 }],
	];
	EFLogEventParser *parser = [[EFLogEventParser alloc] initWithMatches:matches];
	return [super initWithCategory:@"KernelEventMonitor" eventParser:parser];
}

@end
