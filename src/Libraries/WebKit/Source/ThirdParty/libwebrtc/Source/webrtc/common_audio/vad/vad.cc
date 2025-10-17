/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
#include "common_audio/vad/include/vad.h"

#include <memory>

#include "common_audio/vad/include/webrtc_vad.h"
#include "rtc_base/checks.h"

namespace webrtc {

namespace {

class VadImpl final : public Vad {
 public:
  explicit VadImpl(Aggressiveness aggressiveness)
      : handle_(nullptr), aggressiveness_(aggressiveness) {
    Reset();
  }

  ~VadImpl() override { WebRtcVad_Free(handle_); }

  Activity VoiceActivity(const int16_t* audio,
                         size_t num_samples,
                         int sample_rate_hz) override {
    int ret = WebRtcVad_Process(handle_, sample_rate_hz, audio, num_samples);
    switch (ret) {
      case 0:
        return kPassive;
      case 1:
        return kActive;
      default:
        RTC_DCHECK_NOTREACHED() << "WebRtcVad_Process returned an error.";
        return kError;
    }
  }

  void Reset() override {
    if (handle_)
      WebRtcVad_Free(handle_);
    handle_ = WebRtcVad_Create();
    RTC_CHECK(handle_);
    RTC_CHECK_EQ(WebRtcVad_Init(handle_), 0);
    RTC_CHECK_EQ(WebRtcVad_set_mode(handle_, aggressiveness_), 0);
  }

 private:
  VadInst* handle_;
  Aggressiveness aggressiveness_;
};

}  // namespace

std::unique_ptr<Vad> CreateVad(Vad::Aggressiveness aggressiveness) {
  return std::unique_ptr<Vad>(new VadImpl(aggressiveness));
}

}  // namespace webrtc
