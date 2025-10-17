/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#include "test/pc/e2e/analyzer/video/analyzing_video_sinks_helper.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "api/test/pclf/media_configuration.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace webrtc_pc_e2e {
namespace {

using ::testing::Eq;

// Asserts equality of the main fields of the video config. We don't compare
// the full config due to the lack of equality definition for a lot of subtypes.
void AssertConfigsAreEquals(const VideoConfig& actual,
                            const VideoConfig& expected) {
  EXPECT_THAT(actual.stream_label, Eq(expected.stream_label));
  EXPECT_THAT(actual.width, Eq(expected.width));
  EXPECT_THAT(actual.height, Eq(expected.height));
  EXPECT_THAT(actual.fps, Eq(expected.fps));
}

TEST(AnalyzingVideoSinksHelperTest, ConfigsCanBeAdded) {
  VideoConfig config("alice_video", /*width=*/1280, /*height=*/720, /*fps=*/30);

  AnalyzingVideoSinksHelper helper;
  helper.AddConfig("alice", config);

  std::optional<std::pair<std::string, VideoConfig>> registred_config =
      helper.GetPeerAndConfig("alice_video");
  ASSERT_TRUE(registred_config.has_value());
  EXPECT_THAT(registred_config->first, Eq("alice"));
  AssertConfigsAreEquals(registred_config->second, config);
}

TEST(AnalyzingVideoSinksHelperTest, AddingForExistingLabelWillOverwriteValue) {
  VideoConfig config_before("alice_video", /*width=*/1280, /*height=*/720,
                            /*fps=*/30);
  VideoConfig config_after("alice_video", /*width=*/640, /*height=*/360,
                           /*fps=*/15);

  AnalyzingVideoSinksHelper helper;
  helper.AddConfig("alice", config_before);

  std::optional<std::pair<std::string, VideoConfig>> registred_config =
      helper.GetPeerAndConfig("alice_video");
  ASSERT_TRUE(registred_config.has_value());
  EXPECT_THAT(registred_config->first, Eq("alice"));
  AssertConfigsAreEquals(registred_config->second, config_before);

  helper.AddConfig("alice", config_after);

  registred_config = helper.GetPeerAndConfig("alice_video");
  ASSERT_TRUE(registred_config.has_value());
  EXPECT_THAT(registred_config->first, Eq("alice"));
  AssertConfigsAreEquals(registred_config->second, config_after);
}

TEST(AnalyzingVideoSinksHelperTest, ConfigsCanBeRemoved) {
  VideoConfig config("alice_video", /*width=*/1280, /*height=*/720, /*fps=*/30);

  AnalyzingVideoSinksHelper helper;
  helper.AddConfig("alice", config);

  ASSERT_TRUE(helper.GetPeerAndConfig("alice_video").has_value());

  helper.RemoveConfig("alice_video");
  ASSERT_FALSE(helper.GetPeerAndConfig("alice_video").has_value());
}

TEST(AnalyzingVideoSinksHelperTest, RemoveOfNonExistingConfigDontCrash) {
  AnalyzingVideoSinksHelper helper;
  helper.RemoveConfig("alice_video");
}

TEST(AnalyzingVideoSinksHelperTest, ClearRemovesAllConfigs) {
  VideoConfig config1("alice_video", /*width=*/640, /*height=*/360, /*fps=*/30);
  VideoConfig config2("bob_video", /*width=*/640, /*height=*/360, /*fps=*/30);

  AnalyzingVideoSinksHelper helper;
  helper.AddConfig("alice", config1);
  helper.AddConfig("bob", config2);

  ASSERT_TRUE(helper.GetPeerAndConfig("alice_video").has_value());
  ASSERT_TRUE(helper.GetPeerAndConfig("bob_video").has_value());

  helper.Clear();
  ASSERT_FALSE(helper.GetPeerAndConfig("alice_video").has_value());
  ASSERT_FALSE(helper.GetPeerAndConfig("bob_video").has_value());
}

struct TestVideoFrameWriterFactory {
  int closed_writers_count = 0;
  int deleted_writers_count = 0;

  std::unique_ptr<test::VideoFrameWriter> CreateWriter() {
    return std::make_unique<TestVideoFrameWriter>(this);
  }

 private:
  class TestVideoFrameWriter : public test::VideoFrameWriter {
   public:
    explicit TestVideoFrameWriter(TestVideoFrameWriterFactory* factory)
        : factory_(factory) {}
    ~TestVideoFrameWriter() override { factory_->deleted_writers_count++; }

    bool WriteFrame(const VideoFrame& frame) override { return true; }

    void Close() override { factory_->closed_writers_count++; }

   private:
    TestVideoFrameWriterFactory* factory_;
  };
};

TEST(AnalyzingVideoSinksHelperTest, RemovingWritersCloseAndDestroyAllOfThem) {
  TestVideoFrameWriterFactory factory;

  AnalyzingVideoSinksHelper helper;
  test::VideoFrameWriter* writer1 =
      helper.AddVideoWriter(factory.CreateWriter());
  test::VideoFrameWriter* writer2 =
      helper.AddVideoWriter(factory.CreateWriter());

  helper.CloseAndRemoveVideoWriters({writer1, writer2});

  EXPECT_THAT(factory.closed_writers_count, Eq(2));
  EXPECT_THAT(factory.deleted_writers_count, Eq(2));
}

TEST(AnalyzingVideoSinksHelperTest, ClearCloseAndDestroyAllWriters) {
  TestVideoFrameWriterFactory factory;

  AnalyzingVideoSinksHelper helper;
  helper.AddVideoWriter(factory.CreateWriter());
  helper.AddVideoWriter(factory.CreateWriter());

  helper.Clear();

  EXPECT_THAT(factory.closed_writers_count, Eq(2));
  EXPECT_THAT(factory.deleted_writers_count, Eq(2));
}

}  // namespace
}  // namespace webrtc_pc_e2e
}  // namespace webrtc
