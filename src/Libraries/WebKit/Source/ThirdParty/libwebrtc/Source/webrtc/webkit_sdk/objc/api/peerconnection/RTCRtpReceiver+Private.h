/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#import "RTCRtpReceiver.h"

#include "api/rtp_receiver_interface.h"

NS_ASSUME_NONNULL_BEGIN

@class RTCPeerConnectionFactory;

namespace webrtc {

class RtpReceiverDelegateAdapter : public RtpReceiverObserverInterface {
 public:
  RtpReceiverDelegateAdapter(RTCRtpReceiver* receiver);

  void OnFirstPacketReceived(cricket::MediaType media_type) override;

 private:
  __weak RTCRtpReceiver* receiver_;
};

}  // namespace webrtc

@interface RTCRtpReceiver ()

@property(nonatomic, readonly) rtc::scoped_refptr<webrtc::RtpReceiverInterface> nativeRtpReceiver;

/** Initialize an RTCRtpReceiver with a native RtpReceiverInterface. */
- (instancetype)initWithFactory:(RTCPeerConnectionFactory*)factory
              nativeRtpReceiver:(rtc::scoped_refptr<webrtc::RtpReceiverInterface>)nativeRtpReceiver
    NS_DESIGNATED_INITIALIZER;

+ (RTCRtpMediaType)mediaTypeForNativeMediaType:(cricket::MediaType)nativeMediaType;

+ (cricket::MediaType)nativeMediaTypeForMediaType:(RTCRtpMediaType)mediaType;

+ (NSString*)stringForMediaType:(RTCRtpMediaType)mediaType;

@end

NS_ASSUME_NONNULL_END
