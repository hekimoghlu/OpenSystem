/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#ifndef CALL_ADAPTATION_VIDEO_STREAM_INPUT_STATE_PROVIDER_H_
#define CALL_ADAPTATION_VIDEO_STREAM_INPUT_STATE_PROVIDER_H_

#include "call/adaptation/encoder_settings.h"
#include "call/adaptation/video_stream_input_state.h"
#include "rtc_base/synchronization/mutex.h"
#include "video/video_stream_encoder_observer.h"

namespace webrtc {

class VideoStreamInputStateProvider {
 public:
  VideoStreamInputStateProvider(
      VideoStreamEncoderObserver* frame_rate_provider);
  virtual ~VideoStreamInputStateProvider();

  void OnHasInputChanged(bool has_input);
  void OnFrameSizeObserved(int frame_size_pixels);
  void OnEncoderSettingsChanged(EncoderSettings encoder_settings);

  virtual VideoStreamInputState InputState();

 private:
  Mutex mutex_;
  VideoStreamEncoderObserver* const frame_rate_provider_;
  VideoStreamInputState input_state_ RTC_GUARDED_BY(mutex_);
};

}  // namespace webrtc

#endif  // CALL_ADAPTATION_VIDEO_STREAM_INPUT_STATE_PROVIDER_H_
