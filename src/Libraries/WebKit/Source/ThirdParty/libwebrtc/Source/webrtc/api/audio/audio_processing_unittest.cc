/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#include "api/audio/audio_processing.h"

#include <memory>

#include "api/environment/environment_factory.h"
#include "api/make_ref_counted.h"
#include "api/scoped_refptr.h"
#include "modules/audio_processing/include/mock_audio_processing.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

using ::testing::_;
using ::testing::NotNull;

TEST(CustomAudioProcessingTest, ReturnsPassedAudioProcessing) {
  scoped_refptr<AudioProcessing> ap =
      make_ref_counted<test::MockAudioProcessing>();

  std::unique_ptr<AudioProcessingBuilderInterface> builder =
      CustomAudioProcessing(ap);

  ASSERT_THAT(builder, NotNull());
  EXPECT_EQ(builder->Build(CreateEnvironment()), ap);
}

#if GTEST_HAS_DEATH_TEST
TEST(CustomAudioProcessingTest, NullptrAudioProcessingIsUnsupported) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnonnull"
  EXPECT_DEATH(CustomAudioProcessing(nullptr), _);
#pragma clang diagnostic pop
}
#endif

}  // namespace webrtc
