/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
#import "RTCMediaStreamTrack.h"
#import "RTCRtpParameters.h"

NS_ASSUME_NONNULL_BEGIN

/** Represents the media type of the RtpReceiver. */
typedef NS_ENUM(NSInteger, RTCRtpMediaType) {
  RTCRtpMediaTypeAudio,
  RTCRtpMediaTypeVideo,
  RTCRtpMediaTypeData,
};

@class RTCRtpReceiver;

RTC_OBJC_EXPORT
@protocol RTCRtpReceiverDelegate <NSObject>

/** Called when the first RTP packet is received.
 *
 *  Note: Currently if there are multiple RtpReceivers of the same media type,
 *  they will all call OnFirstPacketReceived at once.
 *
 *  For example, if we create three audio receivers, A/B/C, they will listen to
 *  the same signal from the underneath network layer. Whenever the first audio packet
 *  is received, the underneath signal will be fired. All the receivers A/B/C will be
 *  notified and the callback of the receiver's delegate will be called.
 *
 *  The process is the same for video receivers.
 */
- (void)rtpReceiver:(RTCRtpReceiver *)rtpReceiver
    didReceiveFirstPacketForMediaType:(RTCRtpMediaType)mediaType;

@end

RTC_OBJC_EXPORT
@protocol RTCRtpReceiver <NSObject>

/** A unique identifier for this receiver. */
@property(nonatomic, readonly) NSString *receiverId;

/** The currently active RTCRtpParameters, as defined in
 *  https://www.w3.org/TR/webrtc/#idl-def-RTCRtpParameters.
 *
 *  The WebRTC specification only defines RTCRtpParameters in terms of senders,
 *  but this API also applies them to receivers, similar to ORTC:
 *  http://ortc.org/wp-content/uploads/2016/03/ortc.html#rtcrtpparameters*.
 */
@property(nonatomic, readonly) RTCRtpParameters *parameters;

/** The RTCMediaStreamTrack associated with the receiver.
 *  Note: reading this property returns a new instance of
 *  RTCMediaStreamTrack. Use isEqual: instead of == to compare
 *  RTCMediaStreamTrack instances.
 */
@property(nonatomic, readonly, nullable) RTCMediaStreamTrack *track;

/** The delegate for this RtpReceiver. */
@property(nonatomic, weak) id<RTCRtpReceiverDelegate> delegate;

@end

RTC_OBJC_EXPORT
@interface RTCRtpReceiver : NSObject <RTCRtpReceiver>

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
