/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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
#include "api/test/create_video_quality_test_fixture.h"

#include <memory>
#include <utility>

#include "api/fec_controller.h"
#include "api/test/video_quality_test_fixture.h"
#include "video/video_quality_test.h"

namespace webrtc {

std::unique_ptr<VideoQualityTestFixtureInterface>
CreateVideoQualityTestFixture() {
  // By default, we don't override the FEC module, so pass an empty factory.
  return std::make_unique<VideoQualityTest>(nullptr);
}

std::unique_ptr<VideoQualityTestFixtureInterface> CreateVideoQualityTestFixture(
    std::unique_ptr<FecControllerFactoryInterface> fec_controller_factory) {
  auto components =
      std::make_unique<VideoQualityTestFixtureInterface::InjectionComponents>();
  components->fec_controller_factory = std::move(fec_controller_factory);
  return std::make_unique<VideoQualityTest>(std::move(components));
}

std::unique_ptr<VideoQualityTestFixtureInterface> CreateVideoQualityTestFixture(
    std::unique_ptr<VideoQualityTestFixtureInterface::InjectionComponents>
        components) {
  return std::make_unique<VideoQualityTest>(std::move(components));
}

}  // namespace webrtc
