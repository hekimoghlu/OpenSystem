/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
#include "api/audio/builtin_audio_processing_builder.h"

#include "api/audio/audio_processing.h"
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/scoped_refptr.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

using ::testing::NotNull;

TEST(BuiltinAudioProcessingBuilderTest, CreatesWithDefaults) {
  EXPECT_THAT(BuiltinAudioProcessingBuilder().Build(CreateEnvironment()),
              NotNull());
}

TEST(BuiltinAudioProcessingBuilderTest, CreatesWithConfig) {
  const Environment env = CreateEnvironment();
  AudioProcessing::Config config;
  // Change a field to make config different to default one.
  config.gain_controller1.enabled = !config.gain_controller1.enabled;

  scoped_refptr<AudioProcessing> ap1 =
      BuiltinAudioProcessingBuilder(config).Build(env);
  ASSERT_THAT(ap1, NotNull());
  EXPECT_EQ(ap1->GetConfig().gain_controller1.enabled,
            config.gain_controller1.enabled);

  scoped_refptr<AudioProcessing> ap2 =
      BuiltinAudioProcessingBuilder().SetConfig(config).Build(env);
  ASSERT_THAT(ap2, NotNull());
  EXPECT_EQ(ap2->GetConfig().gain_controller1.enabled,
            config.gain_controller1.enabled);
}

}  // namespace webrtc
