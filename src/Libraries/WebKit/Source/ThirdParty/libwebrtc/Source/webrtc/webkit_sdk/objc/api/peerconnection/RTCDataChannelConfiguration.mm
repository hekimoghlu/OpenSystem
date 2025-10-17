/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
#import "RTCDataChannelConfiguration+Private.h"

#import "helpers/NSString+StdString.h"

@implementation RTCDataChannelConfiguration

@synthesize nativeDataChannelInit = _nativeDataChannelInit;

- (BOOL)isOrdered {
  return _nativeDataChannelInit.ordered;
}

- (void)setIsOrdered:(BOOL)isOrdered {
  _nativeDataChannelInit.ordered = isOrdered;
}

- (NSInteger)maxRetransmitTimeMs {
  return self.maxPacketLifeTime;
}

- (void)setMaxRetransmitTimeMs:(NSInteger)maxRetransmitTimeMs {
  self.maxPacketLifeTime = maxRetransmitTimeMs;
}

- (int)maxPacketLifeTime {
  return *_nativeDataChannelInit.maxRetransmitTime;
}

- (void)setMaxPacketLifeTime:(int)maxPacketLifeTime {
  _nativeDataChannelInit.maxRetransmitTime = maxPacketLifeTime;
}

- (int)maxRetransmits {
  if (_nativeDataChannelInit.maxRetransmits) {
    return *_nativeDataChannelInit.maxRetransmits;
  } else {
    return -1;
  }
}

- (void)setMaxRetransmits:(int)maxRetransmits {
  _nativeDataChannelInit.maxRetransmits = maxRetransmits;
}

- (NSString *)protocol {
  return [NSString stringForStdString:_nativeDataChannelInit.protocol];
}

- (void)setProtocol:(NSString *)protocol {
  _nativeDataChannelInit.protocol = [NSString stdStringForString:protocol];
}

- (BOOL)isNegotiated {
  return _nativeDataChannelInit.negotiated;
}

- (void)setIsNegotiated:(BOOL)isNegotiated {
  _nativeDataChannelInit.negotiated = isNegotiated;
}

- (int)streamId {
  return self.channelId;
}

- (void)setStreamId:(int)streamId {
  self.channelId = streamId;
}

- (int)channelId {
  return _nativeDataChannelInit.id;
}

- (void)setChannelId:(int)channelId {
  _nativeDataChannelInit.id = channelId;
}

@end
