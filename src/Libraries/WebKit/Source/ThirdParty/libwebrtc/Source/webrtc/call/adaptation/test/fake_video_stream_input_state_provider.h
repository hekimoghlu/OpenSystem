/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 5, 2023.
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
#ifndef CALL_ADAPTATION_TEST_FAKE_VIDEO_STREAM_INPUT_STATE_PROVIDER_H_
#define CALL_ADAPTATION_TEST_FAKE_VIDEO_STREAM_INPUT_STATE_PROVIDER_H_

#include "call/adaptation/video_stream_input_state_provider.h"

namespace webrtc {

class FakeVideoStreamInputStateProvider : public VideoStreamInputStateProvider {
 public:
  FakeVideoStreamInputStateProvider();
  virtual ~FakeVideoStreamInputStateProvider();

  void SetInputState(int input_pixels, int input_fps, int min_pixels_per_frame);
  VideoStreamInputState InputState() override;

 private:
  VideoStreamInputState fake_input_state_;
};

}  // namespace webrtc

#endif  // CALL_ADAPTATION_TEST_FAKE_VIDEO_STREAM_INPUT_STATE_PROVIDER_H_
