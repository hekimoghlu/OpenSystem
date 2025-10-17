/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#import "RTCAudioSource+Private.h"

#include "rtc_base/checks.h"

@implementation RTCAudioSource {
}

@synthesize volume = _volume;
@synthesize nativeAudioSource = _nativeAudioSource;

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
              nativeAudioSource:
                  (rtc::scoped_refptr<webrtc::AudioSourceInterface>)nativeAudioSource {
  RTC_DCHECK(factory);
  RTC_DCHECK(nativeAudioSource);

  if (self = [super initWithFactory:factory
                  nativeMediaSource:nativeAudioSource
                               type:RTCMediaSourceTypeAudio]) {
    _nativeAudioSource = nativeAudioSource;
  }
  return self;
}

- (instancetype)initWithFactory:(RTCPeerConnectionFactory *)factory
              nativeMediaSource:(rtc::scoped_refptr<webrtc::MediaSourceInterface>)nativeMediaSource
                           type:(RTCMediaSourceType)type {
  RTC_NOTREACHED();
  return nil;
}

- (NSString *)description {
  NSString *stateString = [[self class] stringForState:self.state];
  return [NSString stringWithFormat:@"RTCAudioSource( %p ): %@", self, stateString];
}

- (void)setVolume:(double)volume {
  _volume = volume;
  _nativeAudioSource->SetVolume(volume);
}

@end
