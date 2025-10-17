/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#import "RTCMediaSource+Private.h"

#include "rtc_base/checks.h"

@implementation RTCMediaSource {
  RTCPeerConnectionFactory *_factory;
  RTCMediaSourceType _type;
}

@synthesize nativeMediaSource = _nativeMediaSource;

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
              nativeMediaSource:(rtc::scoped_refptr<webrtc::MediaSourceInterface>)nativeMediaSource
                           type:(RTCMediaSourceType)type {
  RTC_DCHECK(factory);
  RTC_DCHECK(nativeMediaSource);
  if (self = [super init]) {
    _factory = factory;
    _nativeMediaSource = nativeMediaSource;
    _type = type;
  }
  return self;
}

- (RTCSourceState)state {
  return [[self class] sourceStateForNativeState:_nativeMediaSource->state()];
}

#pragma mark - Private

+ (webrtc::MediaSourceInterface::SourceState)nativeSourceStateForState:
    (RTCSourceState)state {
  switch (state) {
    case RTCSourceStateInitializing:
      return webrtc::MediaSourceInterface::kInitializing;
    case RTCSourceStateLive:
      return webrtc::MediaSourceInterface::kLive;
    case RTCSourceStateEnded:
      return webrtc::MediaSourceInterface::kEnded;
    case RTCSourceStateMuted:
      return webrtc::MediaSourceInterface::kMuted;
  }
}

+ (RTCSourceState)sourceStateForNativeState:
    (webrtc::MediaSourceInterface::SourceState)nativeState {
  switch (nativeState) {
    case webrtc::MediaSourceInterface::kInitializing:
      return RTCSourceStateInitializing;
    case webrtc::MediaSourceInterface::kLive:
      return RTCSourceStateLive;
    case webrtc::MediaSourceInterface::kEnded:
      return RTCSourceStateEnded;
    case webrtc::MediaSourceInterface::kMuted:
      return RTCSourceStateMuted;
  }
}

+ (NSString *)stringForState:(RTCSourceState)state {
  switch (state) {
    case RTCSourceStateInitializing:
      return @"Initializing";
    case RTCSourceStateLive:
      return @"Live";
    case RTCSourceStateEnded:
      return @"Ended";
    case RTCSourceStateMuted:
      return @"Muted";
  }
}

@end
