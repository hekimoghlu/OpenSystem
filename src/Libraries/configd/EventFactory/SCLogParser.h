/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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
#import <os/log.h>

@class EFLogEventParser;

os_log_t __log_Spectacles(void);
#define specs_log_err(format, ...)	os_log_error(__log_Spectacles(), format, ##__VA_ARGS__)
#define specs_log_notice(format, ...)	os_log      (__log_Spectacles(), format, ##__VA_ARGS__)
#define specs_log_info(format, ...)	os_log_info (__log_Spectacles(), format, ##__VA_ARGS__)
#define specs_log_debug(format, ...)	os_log_debug(__log_Spectacles(), format, ##__VA_ARGS__)

#define TokenInterfaceName "ifname"

@interface SCLogParser: NSObject
- (instancetype)initWithCategory:(NSString *)category eventParser:(EFLogEventParser *)eventParser;
- (NSData *)createSubsystemIdentifier;
- (NSArray<NSString *> *)addUniqueString:(NSString *)newString toArray:(NSArray<NSString *> *)array;
- (NSArray<NSString *> *)addUniqueStrings:(NSArray<NSString *> *)strings toArray:(NSArray<NSString *> *)array;
- (EFNetworkControlPathEvent *)createInterfaceEventWithLogEvent:(EFLogEvent *)logEvent matchResult:(NSTextCheckingResult *)matchResult;
- (EFNetworkControlPathEvent *)createInterfaceEventWithLogEvent:(EFLogEvent *)logEvent interfaceName:(NSString *)interfaceName;
- (void)addAddress:(NSString *)addressString toInterfaceEvent:(EFNetworkControlPathEvent *)event;
- (BOOL)removeAddress:(NSString *)addressString fromInterfaceEvent:(EFNetworkControlPathEvent *)event;
- (NSString *)substringOfString:(NSString *)matchedString forCaptureGroup:(NSString *)groupName inMatchResult:(NSTextCheckingResult *)result;
- (sa_family_t)getAddressFamilyOfAddress:(NSString *)addressString;
@property (readonly) EFLogEventParser *eventParser;
@property (readonly) NSString *category;
@property (class, readonly, atomic) NSMutableDictionary<NSString *, NSArray<NSString *> *> *interfaceMap;
@end
