/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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

#import "RTCMacros.h"

/**
 * Represents the session description type. This exposes the same types that are
 * in C++, which doesn't include the rollback type that is in the W3C spec.
 */
typedef NS_ENUM(NSInteger, RTCSdpType) {
  RTCSdpTypeOffer,
  RTCSdpTypePrAnswer,
  RTCSdpTypeAnswer,
};

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@interface RTCSessionDescription : NSObject

/** The type of session description. */
@property(nonatomic, readonly) RTCSdpType type;

/** The SDP string representation of this session description. */
@property(nonatomic, readonly) NSString *sdp;

- (instancetype)init NS_UNAVAILABLE;

/** Initialize a session description with a type and SDP string. */
- (instancetype)initWithType:(RTCSdpType)type sdp:(NSString *)sdp NS_DESIGNATED_INITIALIZER;

+ (NSString *)stringForType:(RTCSdpType)type;

+ (RTCSdpType)typeForString:(NSString *)string;

@end

NS_ASSUME_NONNULL_END
