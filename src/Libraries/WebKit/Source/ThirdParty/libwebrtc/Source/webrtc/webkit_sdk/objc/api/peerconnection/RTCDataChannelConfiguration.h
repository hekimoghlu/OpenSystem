/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
#import <AvailabilityMacros.h>
#import <Foundation/Foundation.h>

#import "RTCMacros.h"

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@interface RTCDataChannelConfiguration : NSObject

/** Set to YES if ordered delivery is required. */
@property(nonatomic, assign) BOOL isOrdered;

/** Deprecated. Use maxPacketLifeTime. */
@property(nonatomic, assign) NSInteger maxRetransmitTimeMs DEPRECATED_ATTRIBUTE;

/**
 * Max period in milliseconds in which retransmissions will be sent. After this
 * time, no more retransmissions will be sent. -1 if unset.
 */
@property(nonatomic, assign) int maxPacketLifeTime;

/** The max number of retransmissions. -1 if unset. */
@property(nonatomic, assign) int maxRetransmits;

/** Set to YES if the channel has been externally negotiated and we do not send
 * an in-band signalling in the form of an "open" message.
 */
@property(nonatomic, assign) BOOL isNegotiated;

/** Deprecated. Use channelId. */
@property(nonatomic, assign) int streamId DEPRECATED_ATTRIBUTE;

/** The id of the data channel. */
@property(nonatomic, assign) int channelId;

/** Set by the application and opaque to the WebRTC implementation. */
@property(nonatomic) NSString* protocol;

@end

NS_ASSUME_NONNULL_END
