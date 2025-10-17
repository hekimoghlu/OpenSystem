/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
#import <WebKit/WKFoundation.h>

WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface _WKRemoteObjectInterface : NSObject

+ (instancetype)remoteObjectInterfaceWithProtocol:(Protocol *)protocol;

- (id)initWithProtocol:(Protocol *)protocol identifier:(NSString *)identifier;

@property (readonly, nonatomic) Protocol *protocol;
@property (readonly, nonatomic) NSString *identifier;

- (NSSet *)classesForSelector:(SEL)selector argumentIndex:(NSUInteger)argumentIndex ofReply:(BOOL)ofReply;
- (void)setClasses:(NSSet *)classes forSelector:(SEL)selector argumentIndex:(NSUInteger)argumentIndex ofReply:(BOOL)ofReply;

- (NSSet *)classesForSelector:(SEL)selector argumentIndex:(NSUInteger)argumentIndex WK_API_DEPRECATED_WITH_REPLACEMENT("-classesForSelector:argumentIndex:ofReply:", macos(10.10, 12.0), ios(8.0, 15.0));
- (void)setClasses:(NSSet *)classes forSelector:(SEL)selector argumentIndex:(NSUInteger)argumentIndex WK_API_DEPRECATED_WITH_REPLACEMENT("-setClasses:forSelector:argumentIndex:ofReply:", macos(10.10, 12.0), ios(8.0, 15.0));

@end
