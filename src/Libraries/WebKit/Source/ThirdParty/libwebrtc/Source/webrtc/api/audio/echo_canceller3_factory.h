/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 15, 2021.
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
#ifndef API_AUDIO_ECHO_CANCELLER3_FACTORY_H_
#define API_AUDIO_ECHO_CANCELLER3_FACTORY_H_

#include <memory>

#include "api/audio/echo_canceller3_config.h"
#include "api/audio/echo_control.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class RTC_EXPORT EchoCanceller3Factory : public EchoControlFactory {
 public:
  // Factory producing EchoCanceller3 instances with the default configuration.
  EchoCanceller3Factory();

  // Factory producing EchoCanceller3 instances with the specified
  // configuration.
  explicit EchoCanceller3Factory(const EchoCanceller3Config& config);

  // Creates an EchoCanceller3 with a specified channel count and sampling rate.
  std::unique_ptr<EchoControl> Create(int sample_rate_hz,
                                      int num_render_channels,
                                      int num_capture_channels) override;

 private:
  const EchoCanceller3Config config_;
};
}  // namespace webrtc

#endif  // API_AUDIO_ECHO_CANCELLER3_FACTORY_H_
