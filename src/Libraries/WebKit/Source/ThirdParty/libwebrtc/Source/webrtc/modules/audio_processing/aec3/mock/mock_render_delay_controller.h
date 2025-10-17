/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_RENDER_DELAY_CONTROLLER_H_
#define MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_RENDER_DELAY_CONTROLLER_H_

#include <optional>

#include "api/array_view.h"
#include "modules/audio_processing/aec3/downsampled_render_buffer.h"
#include "modules/audio_processing/aec3/render_delay_controller.h"
#include "test/gmock.h"

namespace webrtc {
namespace test {

class MockRenderDelayController : public RenderDelayController {
 public:
  MockRenderDelayController();
  virtual ~MockRenderDelayController();

  MOCK_METHOD(void, Reset, (bool reset_delay_statistics), (override));
  MOCK_METHOD(void, LogRenderCall, (), (override));
  MOCK_METHOD(std::optional<DelayEstimate>,
              GetDelay,
              (const DownsampledRenderBuffer& render_buffer,
               size_t render_delay_buffer_delay,
               const Block& capture),
              (override));
  MOCK_METHOD(bool, HasClockdrift, (), (const, override));
};

}  // namespace test
}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC3_MOCK_MOCK_RENDER_DELAY_CONTROLLER_H_
