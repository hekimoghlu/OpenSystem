/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#include "modules/rtp_rtcp/source/rtp_generic_frame_descriptor_extension.h"

#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

constexpr uint8_t kDeprecatedFlags = 0x30;

TEST(RtpGenericFrameDescriptorExtensionTest,
     ParseFirstPacketOfIndependenSubFrame) {
  const int kTemporalLayer = 5;
  constexpr uint8_t kRaw[] = {0x80 | kTemporalLayer, 0x49, 0x12, 0x34};
  RtpGenericFrameDescriptor descriptor;

  ASSERT_TRUE(RtpGenericFrameDescriptorExtension00::Parse(kRaw, &descriptor));

  EXPECT_TRUE(descriptor.FirstPacketInSubFrame());
  EXPECT_FALSE(descriptor.LastPacketInSubFrame());

  EXPECT_THAT(descriptor.FrameDependenciesDiffs(), IsEmpty());
  EXPECT_EQ(descriptor.TemporalLayer(), kTemporalLayer);
  EXPECT_EQ(descriptor.SpatialLayersBitmask(), 0x49);
  EXPECT_EQ(descriptor.FrameId(), 0x3412);
}

TEST(RtpGenericFrameDescriptorExtensionTest,
     WriteFirstPacketOfIndependenSubFrame) {
  const int kTemporalLayer = 5;
  uint8_t kRaw[] = {0x80 | kTemporalLayer | kDeprecatedFlags, 0x49, 0x12, 0x34};
  RtpGenericFrameDescriptor descriptor;

  descriptor.SetFirstPacketInSubFrame(true);
  descriptor.SetTemporalLayer(kTemporalLayer);
  descriptor.SetSpatialLayersBitmask(0x49);
  descriptor.SetFrameId(0x3412);

  ASSERT_EQ(RtpGenericFrameDescriptorExtension00::ValueSize(descriptor),
            sizeof(kRaw));
  uint8_t buffer[sizeof(kRaw)];
  EXPECT_TRUE(RtpGenericFrameDescriptorExtension00::Write(buffer, descriptor));

  EXPECT_THAT(buffer, ElementsAreArray(kRaw));
}

TEST(RtpGenericFrameDescriptorExtensionTest, ParseLastPacketOfSubFrame) {
  constexpr uint8_t kRaw[] = {0x40};
  RtpGenericFrameDescriptor descriptor;

  ASSERT_TRUE(RtpGenericFrameDescriptorExtension00::Parse(kRaw, &descriptor));

  EXPECT_FALSE(descriptor.FirstPacketInSubFrame());
  EXPECT_TRUE(descriptor.LastPacketInSubFrame());
}

TEST(RtpGenericFrameDescriptorExtensionTest, WriteLastPacketOfSubFrame) {
  uint8_t kRaw[] = {0x40 | kDeprecatedFlags};

  RtpGenericFrameDescriptor descriptor;
  descriptor.SetLastPacketInSubFrame(true);

  ASSERT_EQ(RtpGenericFrameDescriptorExtension00::ValueSize(descriptor),
            sizeof(kRaw));
  uint8_t buffer[sizeof(kRaw)];
  EXPECT_TRUE(RtpGenericFrameDescriptorExtension00::Write(buffer, descriptor));
  EXPECT_THAT(buffer, ElementsAreArray(kRaw));
}

TEST(RtpGenericFrameDescriptorExtensionTest, ParseMinShortFrameDependencies) {
  constexpr uint16_t kDiff = 1;
  constexpr uint8_t kRaw[] = {0x88, 0x01, 0x00, 0x00, 0x04};
  RtpGenericFrameDescriptor descriptor;

  ASSERT_TRUE(RtpGenericFrameDescriptorExtension00::Parse(kRaw, &descriptor));
  ASSERT_TRUE(descriptor.FirstPacketInSubFrame());
  EXPECT_THAT(descriptor.FrameDependenciesDiffs(), ElementsAre(kDiff));
}

TEST(RtpGenericFrameDescriptorExtensionTest, WriteMinShortFrameDependencies) {
  constexpr uint16_t kDiff = 1;
  uint8_t kRaw[] = {0x88 | kDeprecatedFlags, 0x01, 0x00, 0x00, 0x04};
  RtpGenericFrameDescriptor descriptor;
  descriptor.SetFirstPacketInSubFrame(true);
  descriptor.AddFrameDependencyDiff(kDiff);

  ASSERT_EQ(RtpGenericFrameDescriptorExtension00::ValueSize(descriptor),
            sizeof(kRaw));
  uint8_t buffer[sizeof(kRaw)];
  EXPECT_TRUE(RtpGenericFrameDescriptorExtension00::Write(buffer, descriptor));
  EXPECT_THAT(buffer, ElementsAreArray(kRaw));
}

TEST(RtpGenericFrameDescriptorExtensionTest, ParseMaxShortFrameDependencies) {
  constexpr uint16_t kDiff = 0x3f;
  constexpr uint8_t kRaw[] = {0xb8, 0x01, 0x00, 0x00, 0xfc};
  RtpGenericFrameDescriptor descriptor;

  ASSERT_TRUE(RtpGenericFrameDescriptorExtension00::Parse(kRaw, &descriptor));
  ASSERT_TRUE(descriptor.FirstPacketInSubFrame());
  EXPECT_THAT(descriptor.FrameDependenciesDiffs(), ElementsAre(kDiff));
}

TEST(RtpGenericFrameDescriptorExtensionTest, WriteMaxShortFrameDependencies) {
  constexpr uint16_t kDiff = 0x3f;
  uint8_t kRaw[] = {0x88 | kDeprecatedFlags, 0x01, 0x00, 0x00, 0xfc};
  RtpGenericFrameDescriptor descriptor;
  descriptor.SetFirstPacketInSubFrame(true);
  descriptor.AddFrameDependencyDiff(kDiff);

  ASSERT_EQ(RtpGenericFrameDescriptorExtension00::ValueSize(descriptor),
            sizeof(kRaw));
  uint8_t buffer[sizeof(kRaw)];
  EXPECT_TRUE(RtpGenericFrameDescriptorExtension00::Write(buffer, descriptor));
  EXPECT_THAT(buffer, ElementsAreArray(kRaw));
}

TEST(RtpGenericFrameDescriptorExtensionTest, ParseMinLongFrameDependencies) {
  constexpr uint16_t kDiff = 0x40;
  constexpr uint8_t kRaw[] = {0xb8, 0x01, 0x00, 0x00, 0x02, 0x01};
  RtpGenericFrameDescriptor descriptor;

  ASSERT_TRUE(RtpGenericFrameDescriptorExtension00::Parse(kRaw, &descriptor));
  ASSERT_TRUE(descriptor.FirstPacketInSubFrame());
  EXPECT_THAT(descriptor.FrameDependenciesDiffs(), ElementsAre(kDiff));
}

TEST(RtpGenericFrameDescriptorExtensionTest, WriteMinLongFrameDependencies) {
  constexpr uint16_t kDiff = 0x40;
  uint8_t kRaw[] = {0x88 | kDeprecatedFlags, 0x01, 0x00, 0x00, 0x02, 0x01};
  RtpGenericFrameDescriptor descriptor;
  descriptor.SetFirstPacketInSubFrame(true);
  descriptor.AddFrameDependencyDiff(kDiff);

  ASSERT_EQ(RtpGenericFrameDescriptorExtension00::ValueSize(descriptor),
            sizeof(kRaw));
  uint8_t buffer[sizeof(kRaw)];
  EXPECT_TRUE(RtpGenericFrameDescriptorExtension00::Write(buffer, descriptor));
  EXPECT_THAT(buffer, ElementsAreArray(kRaw));
}

TEST(RtpGenericFrameDescriptorExtensionTest,
     ParseLongFrameDependenciesAsBigEndian) {
  constexpr uint16_t kDiff = 0x7654 >> 2;
  constexpr uint8_t kRaw[] = {0xb8, 0x01, 0x00, 0x00, 0x54 | 0x02, 0x76};
  RtpGenericFrameDescriptor descriptor;

  ASSERT_TRUE(RtpGenericFrameDescriptorExtension00::Parse(kRaw, &descriptor));
  ASSERT_TRUE(descriptor.FirstPacketInSubFrame());
  EXPECT_THAT(descriptor.FrameDependenciesDiffs(), ElementsAre(kDiff));
}

TEST(RtpGenericFrameDescriptorExtensionTest,
     WriteLongFrameDependenciesAsBigEndian) {
  constexpr uint16_t kDiff = 0x7654 >> 2;
  uint8_t kRaw[] = {
      0x88 | kDeprecatedFlags, 0x01, 0x00, 0x00, 0x54 | 0x02, 0x76};
  RtpGenericFrameDescriptor descriptor;
  descriptor.SetFirstPacketInSubFrame(true);
  descriptor.AddFrameDependencyDiff(kDiff);

  ASSERT_EQ(RtpGenericFrameDescriptorExtension00::ValueSize(descriptor),
            sizeof(kRaw));
  uint8_t buffer[sizeof(kRaw)];
  EXPECT_TRUE(RtpGenericFrameDescriptorExtension00::Write(buffer, descriptor));
  EXPECT_THAT(buffer, ElementsAreArray(kRaw));
}

TEST(RtpGenericFrameDescriptorExtensionTest, ParseMaxLongFrameDependencies) {
  constexpr uint16_t kDiff = 0x3fff;
  constexpr uint8_t kRaw[] = {0xb8, 0x01, 0x00, 0x00, 0xfe, 0xff};
  RtpGenericFrameDescriptor descriptor;

  ASSERT_TRUE(RtpGenericFrameDescriptorExtension00::Parse(kRaw, &descriptor));
  ASSERT_TRUE(descriptor.FirstPacketInSubFrame());
  EXPECT_THAT(descriptor.FrameDependenciesDiffs(), ElementsAre(kDiff));
}

TEST(RtpGenericFrameDescriptorExtensionTest, WriteMaxLongFrameDependencies) {
  constexpr uint16_t kDiff = 0x3fff;
  uint8_t kRaw[] = {0x88 | kDeprecatedFlags, 0x01, 0x00, 0x00, 0xfe, 0xff};
  RtpGenericFrameDescriptor descriptor;
  descriptor.SetFirstPacketInSubFrame(true);
  descriptor.AddFrameDependencyDiff(kDiff);

  ASSERT_EQ(RtpGenericFrameDescriptorExtension00::ValueSize(descriptor),
            sizeof(kRaw));
  uint8_t buffer[sizeof(kRaw)];
  EXPECT_TRUE(RtpGenericFrameDescriptorExtension00::Write(buffer, descriptor));
  EXPECT_THAT(buffer, ElementsAreArray(kRaw));
}

TEST(RtpGenericFrameDescriptorExtensionTest, ParseTwoFrameDependencies) {
  constexpr uint16_t kDiff1 = 9;
  constexpr uint16_t kDiff2 = 15;
  constexpr uint8_t kRaw[] = {
      0xb8, 0x01, 0x00, 0x00, (kDiff1 << 2) | 0x01, kDiff2 << 2};
  RtpGenericFrameDescriptor descriptor;

  ASSERT_TRUE(RtpGenericFrameDescriptorExtension00::Parse(kRaw, &descriptor));
  ASSERT_TRUE(descriptor.FirstPacketInSubFrame());
  EXPECT_THAT(descriptor.FrameDependenciesDiffs(), ElementsAre(kDiff1, kDiff2));
}

TEST(RtpGenericFrameDescriptorExtensionTest, WriteTwoFrameDependencies) {
  constexpr uint16_t kDiff1 = 9;
  constexpr uint16_t kDiff2 = 15;
  uint8_t kRaw[] = {0x88 | kDeprecatedFlags, 0x01,       0x00, 0x00,
                    (kDiff1 << 2) | 0x01,    kDiff2 << 2};
  RtpGenericFrameDescriptor descriptor;
  descriptor.SetFirstPacketInSubFrame(true);
  descriptor.AddFrameDependencyDiff(kDiff1);
  descriptor.AddFrameDependencyDiff(kDiff2);

  ASSERT_EQ(RtpGenericFrameDescriptorExtension00::ValueSize(descriptor),
            sizeof(kRaw));
  uint8_t buffer[sizeof(kRaw)];
  EXPECT_TRUE(RtpGenericFrameDescriptorExtension00::Write(buffer, descriptor));
  EXPECT_THAT(buffer, ElementsAreArray(kRaw));
}

TEST(RtpGenericFrameDescriptorExtensionTest,
     ParseResolutionOnIndependentFrame) {
  constexpr int kWidth = 0x2468;
  constexpr int kHeight = 0x6543;
  constexpr uint8_t kRaw[] = {0xb0, 0x01, 0x00, 0x00, 0x24, 0x68, 0x65, 0x43};
  RtpGenericFrameDescriptor descriptor;

  ASSERT_TRUE(RtpGenericFrameDescriptorExtension00::Parse(kRaw, &descriptor));
  EXPECT_EQ(descriptor.Width(), kWidth);
  EXPECT_EQ(descriptor.Height(), kHeight);
}

TEST(RtpGenericFrameDescriptorExtensionTest,
     WriteResolutionOnIndependentFrame) {
  constexpr int kWidth = 0x2468;
  constexpr int kHeight = 0x6543;
  uint8_t kRaw[] = {
      0x80 | kDeprecatedFlags, 0x01, 0x00, 0x00, 0x24, 0x68, 0x65, 0x43};
  RtpGenericFrameDescriptor descriptor;
  descriptor.SetFirstPacketInSubFrame(true);
  descriptor.SetResolution(kWidth, kHeight);

  ASSERT_EQ(RtpGenericFrameDescriptorExtension00::ValueSize(descriptor),
            sizeof(kRaw));
  uint8_t buffer[sizeof(kRaw)];
  EXPECT_TRUE(RtpGenericFrameDescriptorExtension00::Write(buffer, descriptor));
  EXPECT_THAT(buffer, ElementsAreArray(kRaw));
}
}  // namespace
}  // namespace webrtc
