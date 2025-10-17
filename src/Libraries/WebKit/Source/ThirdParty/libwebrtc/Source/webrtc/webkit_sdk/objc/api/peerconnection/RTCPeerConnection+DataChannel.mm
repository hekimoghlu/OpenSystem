/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#import "RTCPeerConnection+Private.h"

#import "RTCDataChannel+Private.h"
#import "RTCDataChannelConfiguration+Private.h"
#import "helpers/NSString+StdString.h"

@implementation RTCPeerConnection (DataChannel)

- (nullable RTCDataChannel *)dataChannelForLabel:(NSString *)label
                                   configuration:(RTCDataChannelConfiguration *)configuration {
  std::string labelString = [NSString stdStringForString:label];
  const webrtc::DataChannelInit nativeInit =
      configuration.nativeDataChannelInit;
  rtc::scoped_refptr<webrtc::DataChannelInterface> dataChannel =
      self.nativePeerConnection->CreateDataChannel(labelString,
                                                   &nativeInit);
  if (!dataChannel) {
    return nil;
  }
  return [[RTCDataChannel alloc] initWithFactory:self.factory nativeDataChannel:dataChannel];
}

@end
