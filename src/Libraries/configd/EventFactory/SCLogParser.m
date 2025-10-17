/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#import <arpa/inet.h>

#import "SCLogParser.h"

@interface SCLogParser ()
@property uint64_t nextEventCounter;
@end

@implementation SCLogParser

- (instancetype)initWithCategory:(NSString *)category eventParser:(EFLogEventParser *)eventParser
{
	self = [super init];
	if (self) {
		_category = category;
		_eventParser = eventParser;
		_nextEventCounter = 1;
	}
	return self;
}

+ (NSMutableDictionary<NSString *, NSArray<NSString *> *> *)interfaceMap
{
	static NSMutableDictionary<NSString *, NSArray<NSString *> *> *_interfaceMap = nil;
	static dispatch_once_t onceToken;
	dispatch_once(&onceToken, ^{
		_interfaceMap = [[NSMutableDictionary alloc] init];
	});
	return _interfaceMap;
}

- (NSData *)createSubsystemIdentifier
{
	const char *utf8String = NULL;
	NSString *identifierString = [[NSString alloc] initWithFormat:@"%@.%llu", _category, _nextEventCounter];
	_nextEventCounter++;
	utf8String = identifierString.UTF8String;
	return [[NSData alloc] initWithBytes:utf8String length:strlen(utf8String)];
}

- (NSArray<NSString *> *)addUniqueString:(NSString *)newString toArray:(NSArray<NSString *> *)array
{
	if (array.count > 0) {
		NSInteger index = [array indexOfObject:newString];
		return (index == NSNotFound ? [array arrayByAddingObject:newString] : array);
	} else {
		return @[ newString ];
	}
}

- (NSArray<NSString *> *)addUniqueStrings:(NSArray<NSString *> *)strings toArray:(NSArray<NSString *> *)array
{
	NSMutableArray<NSString *> *uniqueStrings;

	if (array == nil) {
		return strings;
	}

	uniqueStrings = [[NSMutableArray alloc] init];
	for (NSString *string in strings) {
		NSInteger index = [array indexOfObject:string];
		if (index == NSNotFound) {
			[uniqueStrings addObject:string];
		}
	}
	if (uniqueStrings.count > 0) {
		return [array arrayByAddingObjectsFromArray:uniqueStrings];
	} else {
		return array;
	}
}

- (EFNetworkControlPathEvent *)createInterfaceEventWithLogEvent:(EFLogEvent *)logEvent matchResult:(NSTextCheckingResult *)matchResult
{
	EFNetworkControlPathEvent *newEvent = nil;
	NSString *interfaceName = [logEvent substringForCaptureGroup:@TokenInterfaceName inMatchResult:matchResult];
	if (interfaceName != nil) {
		NSData *identifier = nil;
		if (SCLogParser.interfaceMap[interfaceName] == nil) {
			SCLogParser.interfaceMap[interfaceName] = @[];
		}
		identifier = [self createSubsystemIdentifier];
		newEvent = [[EFNetworkControlPathEvent alloc] initWithLogEvent:logEvent subsystemIdentifier:identifier];
		newEvent.interfaceBSDName = interfaceName;
	}
	return newEvent;
}

- (EFNetworkControlPathEvent *)createInterfaceEventWithLogEvent:(EFLogEvent *)logEvent interfaceName:(NSString *)interfaceName
{
	EFNetworkControlPathEvent *newEvent = nil;
	NSData *identifier = [self createSubsystemIdentifier];
	if (SCLogParser.interfaceMap[interfaceName] == nil) {
		SCLogParser.interfaceMap[interfaceName] = @[];
	}
	newEvent = [[EFNetworkControlPathEvent alloc] initWithLogEvent:logEvent subsystemIdentifier:identifier];
	newEvent.interfaceBSDName = interfaceName;
	return newEvent;
}

- (void)addAddress:(NSString *)addressString toInterfaceEvent:(EFNetworkControlPathEvent *)event
{
	if (event.interfaceBSDName != nil) {
		NSArray<NSString *> *addresses = SCLogParser.interfaceMap[event.interfaceBSDName];
		event.addresses = [self addUniqueString:addressString toArray:addresses];
		SCLogParser.interfaceMap[event.interfaceBSDName] = event.addresses;
	}
}

- (BOOL)removeAddress:(NSString *)addressString fromInterfaceEvent:(EFNetworkControlPathEvent *)event
{
	if (event.interfaceBSDName != nil) {
		NSArray<NSString *> *addresses = SCLogParser.interfaceMap[event.interfaceBSDName];
		if (addresses.count > 0) {
			NSPredicate *matchPredicate = [NSPredicate predicateWithBlock:^BOOL(id evaluatedObject, __unused NSDictionary<NSString *,id> *bindings) {
				NSString *matchString = (NSString *)evaluatedObject;
				if (matchString != nil) {
					return ![matchString isEqualToString:addressString];
				} else {
					return NO;
				}
			}];
			event.addresses = [addresses filteredArrayUsingPredicate:matchPredicate];
			SCLogParser.interfaceMap[event.interfaceBSDName] = event.addresses;
			return YES;
		}
	}

	return NO;
}

- (NSString *)substringOfString:(NSString *)matchedString forCaptureGroup:(NSString *)groupName inMatchResult:(NSTextCheckingResult *)result
{
	NSString *substring = nil;
	NSRange groupRange = [result rangeWithName:groupName];
	if (!NSEqualRanges(groupRange, NSMakeRange(NSNotFound, 0))) {
		substring = [matchedString substringWithRange:groupRange];
	}
	return substring;
}

- (sa_family_t)getAddressFamilyOfAddress:(NSString *)addressString
{
	sa_family_t af = AF_UNSPEC;
	struct in6_addr v6addr;
	struct in_addr v4addr;

	if (inet_pton(AF_INET6, addressString.UTF8String, &v6addr) == 1) {
		af = AF_INET6;
	} else if (inet_pton(AF_INET, addressString.UTF8String, &v4addr)) {
		af = AF_INET;
	}

	return af;
}

@end
