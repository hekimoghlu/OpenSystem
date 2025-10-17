/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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
#ifndef SDK_OBJC_NATIVE_SRC_AUDIO_AUDIO_SESSION_OBSERVER_H_
#define SDK_OBJC_NATIVE_SRC_AUDIO_AUDIO_SESSION_OBSERVER_H_

#include "rtc_base/thread.h"

namespace webrtc {

// Observer interface for listening to AVAudioSession events.
class AudioSessionObserver {
 public:
  // Called when audio session interruption begins.
  virtual void OnInterruptionBegin() = 0;

  // Called when audio session interruption ends.
  virtual void OnInterruptionEnd() = 0;

  // Called when audio route changes.
  virtual void OnValidRouteChange() = 0;

  // Called when the ability to play or record changes.
  virtual void OnCanPlayOrRecordChange(bool can_play_or_record) = 0;

  virtual void OnChangedOutputVolume() = 0;

 protected:
  virtual ~AudioSessionObserver() {}
};

}  // namespace webrtc

#endif  //  SDK_OBJC_NATIVE_SRC_AUDIO_AUDIO_SESSION_OBSERVER_H_
