/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#import "RTCDtmfSender+Private.h"

#import "base/RTCLogging.h"
#import "helpers/NSString+StdString.h"

#include "rtc_base/time_utils.h"

@implementation RTCDtmfSender {
  rtc::scoped_refptr<webrtc::DtmfSenderInterface> _nativeDtmfSender;
}

- (BOOL)canInsertDtmf {
  return _nativeDtmfSender->CanInsertDtmf();
}

- (BOOL)insertDtmf:(nonnull NSString *)tones
          duration:(NSTimeInterval)duration
      interToneGap:(NSTimeInterval)interToneGap {
  RTC_DCHECK(tones != nil);

  int durationMs = static_cast<int>(duration * rtc::kNumMillisecsPerSec);
  int interToneGapMs = static_cast<int>(interToneGap * rtc::kNumMillisecsPerSec);
  return _nativeDtmfSender->InsertDtmf(
      [NSString stdStringForString:tones], durationMs, interToneGapMs);
}

- (nonnull NSString *)remainingTones {
  return [NSString stringForStdString:_nativeDtmfSender->tones()];
}

- (NSTimeInterval)duration {
  return static_cast<NSTimeInterval>(_nativeDtmfSender->duration()) / rtc::kNumMillisecsPerSec;
}

- (NSTimeInterval)interToneGap {
  return static_cast<NSTimeInterval>(_nativeDtmfSender->inter_tone_gap()) /
      rtc::kNumMillisecsPerSec;
}

- (NSString *)description {
  return [NSString
      stringWithFormat:
          @"RTCDtmfSender {\n  remainingTones: %@\n  duration: %f sec\n  interToneGap: %f sec\n}",
          [self remainingTones],
          [self duration],
          [self interToneGap]];
}

#pragma mark - Private

- (rtc::scoped_refptr<webrtc::DtmfSenderInterface>)nativeDtmfSender {
  return _nativeDtmfSender;
}

- (instancetype)initWithNativeDtmfSender:
        (rtc::scoped_refptr<webrtc::DtmfSenderInterface>)nativeDtmfSender {
  NSParameterAssert(nativeDtmfSender);
  if (self = [super init]) {
    _nativeDtmfSender = nativeDtmfSender;
    RTCLogInfo(@"RTCDtmfSender(%p): created DTMF sender: %@", self, self.description);
  }
  return self;
}
@end
