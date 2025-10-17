/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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
#include "common_video/corruption_detection_converters.h"

#include <optional>
#include <vector>

#include "common_video/corruption_detection_message.h"
#include "common_video/frame_instrumentation_data.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::testing::_;
using ::testing::ElementsAre;

TEST(FrameInstrumentationDataToCorruptionDetectionMessageTest,
     ConvertsValidData) {
  FrameInstrumentationData data = {.sequence_index = 1,
                                   .communicate_upper_bits = false,
                                   .std_dev = 1.0,
                                   .luma_error_threshold = 5,
                                   .chroma_error_threshold = 5,
                                   .sample_values = {1.0, 2.0, 3.0, 4.0, 5.0}};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationDataToCorruptionDetectionMessage(data);
  ASSERT_TRUE(message.has_value());
  EXPECT_EQ(message->sequence_index(), 1);
  EXPECT_FALSE(message->interpret_sequence_index_as_most_significant_bits());
  EXPECT_EQ(message->std_dev(), 1.0);
  EXPECT_EQ(message->luma_error_threshold(), 5);
  EXPECT_EQ(message->chroma_error_threshold(), 5);
  EXPECT_THAT(message->sample_values(), ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0));
}

TEST(FrameInstrumentationDataToCorruptionDetectionMessageTest,
     ReturnsNulloptWhenSequenceIndexIsNegative) {
  FrameInstrumentationData data = {.sequence_index = -1,
                                   .communicate_upper_bits = false,
                                   .std_dev = 1.0,
                                   .luma_error_threshold = 5,
                                   .chroma_error_threshold = 5,
                                   .sample_values = {1.0, 2.0, 3.0, 4.0, 5.0}};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationDataToCorruptionDetectionMessage(data);
  ASSERT_FALSE(message.has_value());
}

TEST(FrameInstrumentationDataToCorruptionDetectionMessageTest,
     ReturnsNulloptWhenSequenceIndexIsTooLarge) {
  // Sequence index must be at max 14 bits.
  FrameInstrumentationData data = {.sequence_index = 0x4000,
                                   .communicate_upper_bits = false,
                                   .std_dev = 1.0,
                                   .luma_error_threshold = 5,
                                   .chroma_error_threshold = 5,
                                   .sample_values = {1.0, 2.0, 3.0, 4.0, 5.0}};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationDataToCorruptionDetectionMessage(data);
  ASSERT_FALSE(message.has_value());
}

TEST(FrameInstrumentationDataToCorruptionDetectionMessageTest,
     ReturnsNulloptWhenThereAreNoSampleValues) {
  // FrameInstrumentationData must by definition have at least one sample value.
  FrameInstrumentationData data = {.sequence_index = 1,
                                   .communicate_upper_bits = false,
                                   .std_dev = 1.0,
                                   .luma_error_threshold = 5,
                                   .chroma_error_threshold = 5,
                                   .sample_values = {}};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationDataToCorruptionDetectionMessage(data);
  ASSERT_FALSE(message.has_value());
}

TEST(FrameInstrumentationDataToCorruptionDetectionMessageTest,
     ReturnsNulloptWhenNotSpecifyingSampleValues) {
  FrameInstrumentationData data = {.sequence_index = 1,
                                   .communicate_upper_bits = false,
                                   .std_dev = 1.0,
                                   .luma_error_threshold = 5,
                                   .chroma_error_threshold = 5};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationDataToCorruptionDetectionMessage(data);
  ASSERT_FALSE(message.has_value());
}

TEST(FrameInstrumentationDataToCorruptionDetectionMessageTest,
     ConvertsSequenceIndexWhenSetToUseUpperBits) {
  FrameInstrumentationData data = {.sequence_index = 0b0000'0110'0000'0101,
                                   .communicate_upper_bits = true,
                                   .std_dev = 1.0,
                                   .luma_error_threshold = 5,
                                   .chroma_error_threshold = 5,
                                   .sample_values = {1.0, 2.0, 3.0, 4.0, 5.0}};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationDataToCorruptionDetectionMessage(data);
  ASSERT_TRUE(message.has_value());
  EXPECT_EQ(message->sequence_index(), 0b0000'1100);
  EXPECT_TRUE(message->interpret_sequence_index_as_most_significant_bits());
  EXPECT_EQ(message->std_dev(), 1.0);
  EXPECT_EQ(message->luma_error_threshold(), 5);
  EXPECT_EQ(message->chroma_error_threshold(), 5);
  EXPECT_THAT(message->sample_values(), ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0));
}

TEST(FrameInstrumentationDataToCorruptionDetectionMessageTest,
     ConvertsSequenceIndexWhenSetToUseLowerBits) {
  FrameInstrumentationData data = {.sequence_index = 0b0000'0110'0000'0101,
                                   .communicate_upper_bits = false,
                                   .std_dev = 1.0,
                                   .luma_error_threshold = 5,
                                   .chroma_error_threshold = 5,
                                   .sample_values = {1.0, 2.0, 3.0, 4.0, 5.0}};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationDataToCorruptionDetectionMessage(data);
  ASSERT_TRUE(message.has_value());
  EXPECT_EQ(message->sequence_index(), 0b0000'0101);
  EXPECT_FALSE(message->interpret_sequence_index_as_most_significant_bits());
  EXPECT_EQ(message->std_dev(), 1.0);
  EXPECT_EQ(message->luma_error_threshold(), 5);
  EXPECT_EQ(message->chroma_error_threshold(), 5);
  EXPECT_THAT(message->sample_values(), ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0));
}

TEST(FrameInstrumentationSyncDataToCorruptionDetectionMessageTest,
     ConvertsValidSyncData) {
  FrameInstrumentationSyncData data = {.sequence_index = 1,
                                       .communicate_upper_bits = true};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationSyncDataToCorruptionDetectionMessage(data);
  ASSERT_TRUE(message.has_value());
  EXPECT_EQ(message->sequence_index(), 0);
  EXPECT_TRUE(message->interpret_sequence_index_as_most_significant_bits());
}

#if GTEST_HAS_DEATH_TEST && RTC_DCHECK_IS_ON
TEST(FrameInstrumentationSyncDataToCorruptionDetectionMessageTest,
     FailsWhenSetToNotCommunicateUpperBits) {
  FrameInstrumentationSyncData data = {.sequence_index = 1,
                                       .communicate_upper_bits = false};

  EXPECT_DEATH(
      ConvertFrameInstrumentationSyncDataToCorruptionDetectionMessage(data), _);
}
#endif  // GTEST_HAS_DEATH_TEST

TEST(FrameInstrumentationSyncDataToCorruptionDetectionMessageTest,
     ReturnsNulloptWhenSyncSequenceIndexIsNegative) {
  FrameInstrumentationSyncData data = {.sequence_index = -1,
                                       .communicate_upper_bits = true};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationSyncDataToCorruptionDetectionMessage(data);
  ASSERT_FALSE(message.has_value());
}

TEST(FrameInstrumentationSyncDataToCorruptionDetectionMessageTest,
     ReturnsNulloptWhenSyncSequenceIndexIsTooLarge) {
  FrameInstrumentationSyncData data = {.sequence_index = 0x4000,
                                       .communicate_upper_bits = true};

  std::optional<CorruptionDetectionMessage> message =
      ConvertFrameInstrumentationSyncDataToCorruptionDetectionMessage(data);
  ASSERT_FALSE(message.has_value());
}

TEST(CorruptionDetectionMessageToFrameInstrumentationData,
     FailWhenPreviousSequenceIndexIsNegative) {
  std::vector<double> sample_values = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithSampleValues(sample_values)
          .Build();
  ASSERT_TRUE(message.has_value());

  EXPECT_FALSE(
      ConvertCorruptionDetectionMessageToFrameInstrumentationData(*message, -1)
          .has_value());
}

TEST(CorruptionDetectionMessageToFrameInstrumentationData,
     FailWhenNoSampleValuesAreProvided) {
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder().Build();
  ASSERT_TRUE(message.has_value());

  EXPECT_FALSE(
      ConvertCorruptionDetectionMessageToFrameInstrumentationData(*message, 0)
          .has_value());
}

TEST(CorruptionDetectionMessageToFrameInstrumentationData,
     IgnorePreviousSequenceIndexWhenSetToUpdateTheMostSignificantBits) {
  std::vector<double> sample_values = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithSequenceIndex(11)
          .WithInterpretSequenceIndexAsMostSignificantBits(true)
          .WithSampleValues(sample_values)
          .Build();
  ASSERT_TRUE(message.has_value());

  std::optional<FrameInstrumentationData> data =
      ConvertCorruptionDetectionMessageToFrameInstrumentationData(*message, 12);

  ASSERT_TRUE(data.has_value());
  EXPECT_EQ(data->sequence_index, 0b0101'1000'0000);
}

TEST(CorruptionDetectionMessageToFrameInstrumentationData,
     UseMessageSequenceIndexWhenHigherThanPrevious) {
  std::vector<double> sample_values = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithSequenceIndex(11)
          .WithInterpretSequenceIndexAsMostSignificantBits(false)
          .WithSampleValues(sample_values)
          .Build();
  ASSERT_TRUE(message.has_value());

  std::optional<FrameInstrumentationData> data =
      ConvertCorruptionDetectionMessageToFrameInstrumentationData(*message, 0);

  ASSERT_TRUE(data.has_value());
  EXPECT_EQ(data->sequence_index, 11);
}

TEST(CorruptionDetectionMessageToFrameInstrumentationData,
     IncreaseThePreviousIdxUntilLsbsAreEqualToTheUpdateWhenTheUpdateIsLsbs) {
  std::vector<double> sample_values = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithSequenceIndex(11)
          .WithInterpretSequenceIndexAsMostSignificantBits(false)
          .WithSampleValues(sample_values)
          .Build();
  ASSERT_TRUE(message.has_value());

  std::optional<FrameInstrumentationData> data =
      ConvertCorruptionDetectionMessageToFrameInstrumentationData(*message,
                                                                  1 + 128);

  ASSERT_TRUE(data.has_value());
  EXPECT_EQ(data->sequence_index, 11 + 128);
}

TEST(CorruptionDetectionMessageToFrameInstrumentationData,
     IgnoreIndexUpdateWhenTheLowerBitsSuppliedAreTheSameAsInThePreviousIndex) {
  std::vector<double> sample_values = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithSequenceIndex(11)
          .WithInterpretSequenceIndexAsMostSignificantBits(false)
          .WithSampleValues(sample_values)
          .Build();
  ASSERT_TRUE(message.has_value());

  std::optional<FrameInstrumentationData> data =
      ConvertCorruptionDetectionMessageToFrameInstrumentationData(*message,
                                                                  11 + 128);

  ASSERT_TRUE(data.has_value());
  EXPECT_EQ(data->sequence_index, 11 + 128);
}

TEST(
    CorruptionDetectionMessageToFrameInstrumentationData,
    IncreaseTheMsbsByOneAndSetTheMessagesLsbWhenMessageLsbIsLowerThanPrevious) {
  std::vector<double> sample_values = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithSequenceIndex(11)
          .WithInterpretSequenceIndexAsMostSignificantBits(false)
          .WithSampleValues(sample_values)
          .Build();
  ASSERT_TRUE(message.has_value());

  std::optional<FrameInstrumentationData> data =
      ConvertCorruptionDetectionMessageToFrameInstrumentationData(*message, 12);

  ASSERT_TRUE(data.has_value());
  EXPECT_EQ(data->sequence_index, 11 + 128);
}

TEST(CorruptionDetectionMessageToFrameInstrumentationData, ConvertAllFields) {
  std::vector<double> sample_values = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithSequenceIndex(11)
          .WithInterpretSequenceIndexAsMostSignificantBits(false)
          .WithStdDev(1.2)
          .WithLumaErrorThreshold(10)
          .WithChromaErrorThreshold(10)
          .WithSampleValues(sample_values)
          .Build();
  ASSERT_TRUE(message.has_value());

  std::optional<FrameInstrumentationData> data =
      ConvertCorruptionDetectionMessageToFrameInstrumentationData(*message, 0);

  ASSERT_TRUE(data.has_value());
  EXPECT_EQ(data->sequence_index, 11);
  EXPECT_FALSE(data->communicate_upper_bits);
  EXPECT_NEAR(data->std_dev, 1.2, 0.024);  // ~2%
  EXPECT_EQ(data->luma_error_threshold, 10);
  EXPECT_EQ(data->chroma_error_threshold, 10);
  EXPECT_THAT(data->sample_values, ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0));
}

TEST(CorruptionDetectionMessageToFrameInstrumentationSyncData,
     FailWhenPreviousSequenceIndexIsNegative) {
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithInterpretSequenceIndexAsMostSignificantBits(true)
          .Build();
  ASSERT_TRUE(message.has_value());

  EXPECT_FALSE(ConvertCorruptionDetectionMessageToFrameInstrumentationSyncData(
                   *message, -1)
                   .has_value());
}

TEST(CorruptionDetectionMessageToFrameInstrumentationSyncData,
     FailWhenSampleValuesArePresent) {
  std::vector<double> sample_values = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithInterpretSequenceIndexAsMostSignificantBits(true)
          .WithSampleValues(sample_values)
          .Build();
  ASSERT_TRUE(message.has_value());

  EXPECT_FALSE(ConvertCorruptionDetectionMessageToFrameInstrumentationSyncData(
                   *message, 0)
                   .has_value());
}

TEST(CorruptionDetectionMessageToFrameInstrumentationSyncData,
     FailWhenSetToUpdateTheLowerBits) {
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithInterpretSequenceIndexAsMostSignificantBits(false)
          .Build();
  ASSERT_TRUE(message.has_value());

  EXPECT_FALSE(ConvertCorruptionDetectionMessageToFrameInstrumentationSyncData(
                   *message, 0)
                   .has_value());
}

TEST(CorruptionDetectionMessageToFrameInstrumentationSyncData,
     IgnorePreviousSequenceIndex) {
  std::optional<CorruptionDetectionMessage> message =
      CorruptionDetectionMessage::Builder()
          .WithSequenceIndex(11)
          .WithInterpretSequenceIndexAsMostSignificantBits(true)
          .Build();
  ASSERT_TRUE(message.has_value());

  std::optional<FrameInstrumentationSyncData> data =
      ConvertCorruptionDetectionMessageToFrameInstrumentationSyncData(*message,
                                                                      12);

  ASSERT_TRUE(data.has_value());
  EXPECT_EQ(data->sequence_index, 0b0101'1000'0000);
  EXPECT_TRUE(data->communicate_upper_bits);
}

}  // namespace
}  // namespace webrtc
