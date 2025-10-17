/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#import "RTCNativeAudioSessionDelegateAdapter.h"

#include "sdk/objc/native/src/audio/audio_session_observer.h"

#import "base/RTCLogging.h"

@implementation RTCNativeAudioSessionDelegateAdapter {
  webrtc::AudioSessionObserver *_observer;
}

- (instancetype)initWithObserver:(webrtc::AudioSessionObserver *)observer {
  RTC_DCHECK(observer);
  if (self = [super init]) {
    _observer = observer;
  }
  return self;
}

#pragma mark - RTCAudioSessionDelegate

- (void)audioSessionDidBeginInterruption:(RTCAudioSession *)session {
  _observer->OnInterruptionBegin();
}

- (void)audioSessionDidEndInterruption:(RTCAudioSession *)session
                   shouldResumeSession:(BOOL)shouldResumeSession {
  _observer->OnInterruptionEnd();
}

- (void)audioSessionDidChangeRoute:(RTCAudioSession *)session
           reason:(AVAudioSessionRouteChangeReason)reason
    previousRoute:(AVAudioSessionRouteDescription *)previousRoute {
  switch (reason) {
    case AVAudioSessionRouteChangeReasonUnknown:
    case AVAudioSessionRouteChangeReasonNewDeviceAvailable:
    case AVAudioSessionRouteChangeReasonOldDeviceUnavailable:
    case AVAudioSessionRouteChangeReasonCategoryChange:
      // It turns out that we see a category change (at least in iOS 9.2)
      // when making a switch from a BT device to e.g. Speaker using the
      // iOS Control Center and that we therefore must check if the sample
      // rate has changed. And if so is the case, restart the audio unit.
    case AVAudioSessionRouteChangeReasonOverride:
    case AVAudioSessionRouteChangeReasonWakeFromSleep:
    case AVAudioSessionRouteChangeReasonNoSuitableRouteForCategory:
      _observer->OnValidRouteChange();
      break;
    case AVAudioSessionRouteChangeReasonRouteConfigurationChange:
      // The set of input and output ports has not changed, but their
      // configuration has, e.g., a portâ€™s selected data source has
      // changed. Ignore this type of route change since we are focusing
      // on detecting headset changes.
      RTCLog(@"Ignoring RouteConfigurationChange");
      break;
  }
}

- (void)audioSessionMediaServerTerminated:(RTCAudioSession *)session {
}

- (void)audioSessionMediaServerReset:(RTCAudioSession *)session {
}

- (void)audioSession:(RTCAudioSession *)session
    didChangeCanPlayOrRecord:(BOOL)canPlayOrRecord {
  _observer->OnCanPlayOrRecordChange(canPlayOrRecord);
}

- (void)audioSessionDidStartPlayOrRecord:(RTCAudioSession *)session {
}

- (void)audioSessionDidStopPlayOrRecord:(RTCAudioSession *)session {
}

- (void)audioSession:(RTCAudioSession *)audioSession
    didChangeOutputVolume:(float)outputVolume {
  _observer->OnChangedOutputVolume();
}

@end
