/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@interface RTC_OBJC_TYPE (RTCIceCandidateErrorEvent) : NSObject

/** The local IP address used to communicate with the STUN or TURN server. */
@property(nonatomic, readonly) NSString *address;

/** The port used to communicate with the STUN or TURN server. */
@property(nonatomic, readonly) int port;

/** The STUN or TURN URL that identifies the STUN or TURN server for which the failure occurred. */
@property(nonatomic, readonly) NSString *url;

/** The numeric STUN error code returned by the STUN or TURN server. If no host candidate can reach
 * the server, errorCode will be set to the value 701 which is outside the STUN error code range.
 * This error is only fired once per server URL while in the RTCIceGatheringState of "gathering". */
@property(nonatomic, readonly) int errorCode;

/** The STUN reason text returned by the STUN or TURN server. If the server could not be reached,
 * errorText will be set to an implementation-specific value providing details about the error. */
@property(nonatomic, readonly) NSString *errorText;

- (instancetype)init NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
