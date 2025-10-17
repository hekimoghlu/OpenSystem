/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#include "pc/session_description.h"

#include "test/gtest.h"

namespace cricket {

TEST(MediaContentDescriptionTest, ExtmapAllowMixedDefaultValue) {
  VideoContentDescription video_desc;
  EXPECT_EQ(MediaContentDescription::kMedia,
            video_desc.extmap_allow_mixed_enum());
}

TEST(MediaContentDescriptionTest, SetExtmapAllowMixed) {
  VideoContentDescription video_desc;
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kNo);
  EXPECT_EQ(MediaContentDescription::kNo, video_desc.extmap_allow_mixed_enum());
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kMedia);
  EXPECT_EQ(MediaContentDescription::kMedia,
            video_desc.extmap_allow_mixed_enum());
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kSession);
  EXPECT_EQ(MediaContentDescription::kSession,
            video_desc.extmap_allow_mixed_enum());

  // Not allowed to downgrade from kSession to kMedia.
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kMedia);
  EXPECT_EQ(MediaContentDescription::kSession,
            video_desc.extmap_allow_mixed_enum());

  // Always okay to set not supported.
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kNo);
  EXPECT_EQ(MediaContentDescription::kNo, video_desc.extmap_allow_mixed_enum());
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kMedia);
  EXPECT_EQ(MediaContentDescription::kMedia,
            video_desc.extmap_allow_mixed_enum());
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kNo);
  EXPECT_EQ(MediaContentDescription::kNo, video_desc.extmap_allow_mixed_enum());
}

TEST(MediaContentDescriptionTest, MixedOneTwoByteHeaderSupported) {
  VideoContentDescription video_desc;
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kNo);
  EXPECT_FALSE(video_desc.extmap_allow_mixed());
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kMedia);
  EXPECT_TRUE(video_desc.extmap_allow_mixed());
  video_desc.set_extmap_allow_mixed_enum(MediaContentDescription::kSession);
  EXPECT_TRUE(video_desc.extmap_allow_mixed());
}

TEST(SessionDescriptionTest, SetExtmapAllowMixed) {
  SessionDescription session_desc;
  session_desc.set_extmap_allow_mixed(true);
  EXPECT_TRUE(session_desc.extmap_allow_mixed());
  session_desc.set_extmap_allow_mixed(false);
  EXPECT_FALSE(session_desc.extmap_allow_mixed());
}

TEST(SessionDescriptionTest, SetExtmapAllowMixedPropagatesToMediaLevel) {
  SessionDescription session_desc;
  session_desc.AddContent("video", MediaProtocolType::kRtp,
                          std::make_unique<VideoContentDescription>());
  MediaContentDescription* video_desc =
      session_desc.GetContentDescriptionByName("video");

  // Setting true on session level propagates to media level.
  session_desc.set_extmap_allow_mixed(true);
  EXPECT_EQ(MediaContentDescription::kSession,
            video_desc->extmap_allow_mixed_enum());

  // Don't downgrade from session level to media level
  video_desc->set_extmap_allow_mixed_enum(MediaContentDescription::kMedia);
  EXPECT_EQ(MediaContentDescription::kSession,
            video_desc->extmap_allow_mixed_enum());

  // Setting false on session level propagates to media level if the current
  // state is kSession.
  session_desc.set_extmap_allow_mixed(false);
  EXPECT_EQ(MediaContentDescription::kNo,
            video_desc->extmap_allow_mixed_enum());

  // Now possible to set at media level.
  video_desc->set_extmap_allow_mixed_enum(MediaContentDescription::kMedia);
  EXPECT_EQ(MediaContentDescription::kMedia,
            video_desc->extmap_allow_mixed_enum());

  // Setting false on session level does not override on media level if current
  // state is kMedia.
  session_desc.set_extmap_allow_mixed(false);
  EXPECT_EQ(MediaContentDescription::kMedia,
            video_desc->extmap_allow_mixed_enum());

  // Setting true on session level overrides setting on media level.
  session_desc.set_extmap_allow_mixed(true);
  EXPECT_EQ(MediaContentDescription::kSession,
            video_desc->extmap_allow_mixed_enum());
}

TEST(SessionDescriptionTest, AddContentTransfersExtmapAllowMixedSetting) {
  SessionDescription session_desc;
  session_desc.set_extmap_allow_mixed(false);
  std::unique_ptr<MediaContentDescription> audio_desc =
      std::make_unique<AudioContentDescription>();
  audio_desc->set_extmap_allow_mixed_enum(MediaContentDescription::kMedia);

  // If session setting is false, media level setting is preserved when new
  // content is added.
  session_desc.AddContent("audio", MediaProtocolType::kRtp,
                          std::move(audio_desc));
  EXPECT_EQ(MediaContentDescription::kMedia,
            session_desc.GetContentDescriptionByName("audio")
                ->extmap_allow_mixed_enum());

  // If session setting is true, it's transferred to media level when new
  // content is added.
  session_desc.set_extmap_allow_mixed(true);
  std::unique_ptr<MediaContentDescription> video_desc =
      std::make_unique<VideoContentDescription>();
  session_desc.AddContent("video", MediaProtocolType::kRtp,
                          std::move(video_desc));
  EXPECT_EQ(MediaContentDescription::kSession,
            session_desc.GetContentDescriptionByName("video")
                ->extmap_allow_mixed_enum());
}

}  // namespace cricket
