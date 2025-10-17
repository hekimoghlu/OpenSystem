/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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

#import "RTCDtmfSender.h"
#import "RTCMacros.h"
#import "RTCMediaStreamTrack.h"
#import "RTCRtpParameters.h"

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT
@protocol RTCRtpSender <NSObject>

/** A unique identifier for this sender. */
@property(nonatomic, readonly) NSString *senderId;

/** The currently active RTCRtpParameters, as defined in
 *  https://www.w3.org/TR/webrtc/#idl-def-RTCRtpParameters.
 */
@property(nonatomic, copy) RTCRtpParameters *parameters;

/** The RTCMediaStreamTrack associated with the sender.
 *  Note: reading this property returns a new instance of
 *  RTCMediaStreamTrack. Use isEqual: instead of == to compare
 *  RTCMediaStreamTrack instances.
 */
@property(nonatomic, copy, nullable) RTCMediaStreamTrack *track;

/** IDs of streams associated with the RTP sender */
@property(nonatomic, copy) NSArray<NSString *> *streamIds;

/** The RTCDtmfSender accociated with the RTP sender. */
@property(nonatomic, readonly, nullable) id<RTCDtmfSender> dtmfSender;

@end

RTC_OBJC_EXPORT
@interface RTCRtpSender : NSObject <RTCRtpSender>

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
