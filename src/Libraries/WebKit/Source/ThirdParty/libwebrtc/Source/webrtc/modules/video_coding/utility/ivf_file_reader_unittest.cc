/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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
#include "modules/video_coding/utility/ivf_file_reader.h"

#include <memory>
#include <string>

#include "modules/video_coding/utility/ivf_file_writer.h"
#include "test/gtest.h"
#include "test/testsupport/file_utils.h"

namespace webrtc {
namespace {

constexpr int kWidth = 320;
constexpr int kHeight = 240;
constexpr int kNumFrames = 3;
constexpr uint8_t kDummyPayload[4] = {'0', '1', '2', '3'};

}  // namespace

class IvfFileReaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    file_name_ =
        webrtc::test::TempFilename(webrtc::test::OutputPath(), "test_file.ivf");
  }
  void TearDown() override { webrtc::test::RemoveFile(file_name_); }

  bool WriteDummyTestFrames(IvfFileWriter* file_writer,
                            VideoCodecType codec_type,
                            int width,
                            int height,
                            int num_frames,
                            bool use_capture_tims_ms,
                            int spatial_layers_count) {
    EncodedImage frame;
    frame.SetSpatialIndex(spatial_layers_count);
    rtc::scoped_refptr<EncodedImageBuffer> payload = EncodedImageBuffer::Create(
        sizeof(kDummyPayload) * spatial_layers_count);
    for (int i = 0; i < spatial_layers_count; ++i) {
      memcpy(&payload->data()[i * sizeof(kDummyPayload)], kDummyPayload,
             sizeof(kDummyPayload));
      frame.SetSpatialLayerFrameSize(i, sizeof(kDummyPayload));
    }
    frame.SetEncodedData(payload);
    frame._encodedWidth = width;
    frame._encodedHeight = height;
    for (int i = 1; i <= num_frames; ++i) {
      if (use_capture_tims_ms) {
        frame.capture_time_ms_ = i;
      } else {
        frame.SetRtpTimestamp(i);
      }
      if (!file_writer->WriteFrame(frame, codec_type))
        return false;
    }
    return true;
  }

  void CreateTestFile(VideoCodecType codec_type,
                      bool use_capture_tims_ms,
                      int spatial_layers_count) {
    std::unique_ptr<IvfFileWriter> file_writer =
        IvfFileWriter::Wrap(FileWrapper::OpenWriteOnly(file_name_), 0);
    ASSERT_TRUE(file_writer.get());
    ASSERT_TRUE(WriteDummyTestFrames(file_writer.get(), codec_type, kWidth,
                                     kHeight, kNumFrames, use_capture_tims_ms,
                                     spatial_layers_count));
    ASSERT_TRUE(file_writer->Close());
  }

  void ValidateFrame(std::optional<EncodedImage> frame,
                     int frame_index,
                     bool use_capture_tims_ms,
                     int spatial_layers_count) {
    ASSERT_TRUE(frame);
    EXPECT_EQ(frame->SpatialIndex(), spatial_layers_count - 1);
    if (use_capture_tims_ms) {
      EXPECT_EQ(frame->capture_time_ms_, static_cast<int64_t>(frame_index));
      EXPECT_EQ(frame->RtpTimestamp(), static_cast<int64_t>(90 * frame_index));
    } else {
      EXPECT_EQ(frame->RtpTimestamp(), static_cast<int64_t>(frame_index));
    }
    ASSERT_EQ(frame->size(), sizeof(kDummyPayload) * spatial_layers_count);
    for (int i = 0; i < spatial_layers_count; ++i) {
      EXPECT_EQ(memcmp(&frame->data()[i * sizeof(kDummyPayload)], kDummyPayload,
                       sizeof(kDummyPayload)),
                0)
          << std::string(reinterpret_cast<char const*>(
                             &frame->data()[i * sizeof(kDummyPayload)]),
                         sizeof(kDummyPayload));
    }
  }

  void ValidateContent(VideoCodecType codec_type,
                       bool use_capture_tims_ms,
                       int spatial_layers_count) {
    std::unique_ptr<IvfFileReader> reader =
        IvfFileReader::Create(FileWrapper::OpenReadOnly(file_name_));
    ASSERT_TRUE(reader.get());
    EXPECT_EQ(reader->GetVideoCodecType(), codec_type);
    EXPECT_EQ(reader->GetFramesCount(),
              spatial_layers_count * static_cast<size_t>(kNumFrames));
    for (int i = 1; i <= kNumFrames; ++i) {
      ASSERT_TRUE(reader->HasMoreFrames());
      ValidateFrame(reader->NextFrame(), i, use_capture_tims_ms,
                    spatial_layers_count);
      EXPECT_FALSE(reader->HasError());
    }
    EXPECT_FALSE(reader->HasMoreFrames());
    EXPECT_FALSE(reader->NextFrame());
    EXPECT_FALSE(reader->HasError());
    ASSERT_TRUE(reader->Close());
  }

  std::string file_name_;
};

TEST_F(IvfFileReaderTest, BasicVp8FileNtpTimestamp) {
  CreateTestFile(kVideoCodecVP8, false, 1);
  ValidateContent(kVideoCodecVP8, false, 1);
}

TEST_F(IvfFileReaderTest, BasicVP8FileMsTimestamp) {
  CreateTestFile(kVideoCodecVP8, true, 1);
  ValidateContent(kVideoCodecVP8, true, 1);
}

TEST_F(IvfFileReaderTest, BasicVP9FileNtpTimestamp) {
  CreateTestFile(kVideoCodecVP9, false, 1);
  ValidateContent(kVideoCodecVP9, false, 1);
}

TEST_F(IvfFileReaderTest, BasicVP9FileMsTimestamp) {
  CreateTestFile(kVideoCodecVP9, true, 1);
  ValidateContent(kVideoCodecVP9, true, 1);
}

TEST_F(IvfFileReaderTest, BasicAv1FileNtpTimestamp) {
  CreateTestFile(kVideoCodecAV1, false, 1);
  ValidateContent(kVideoCodecAV1, false, 1);
}

TEST_F(IvfFileReaderTest, BasicAv1FileMsTimestamp) {
  CreateTestFile(kVideoCodecAV1, true, 1);
  ValidateContent(kVideoCodecAV1, true, 1);
}

TEST_F(IvfFileReaderTest, BasicH264FileNtpTimestamp) {
  CreateTestFile(kVideoCodecH264, false, 1);
  ValidateContent(kVideoCodecH264, false, 1);
}

TEST_F(IvfFileReaderTest, BasicH264FileMsTimestamp) {
  CreateTestFile(kVideoCodecH264, true, 1);
  ValidateContent(kVideoCodecH264, true, 1);
}

TEST_F(IvfFileReaderTest, MultilayerVp8FileNtpTimestamp) {
  CreateTestFile(kVideoCodecVP8, false, 3);
  ValidateContent(kVideoCodecVP8, false, 3);
}

TEST_F(IvfFileReaderTest, MultilayerVP9FileNtpTimestamp) {
  CreateTestFile(kVideoCodecVP9, false, 3);
  ValidateContent(kVideoCodecVP9, false, 3);
}

TEST_F(IvfFileReaderTest, MultilayerAv1FileNtpTimestamp) {
  CreateTestFile(kVideoCodecAV1, false, 3);
  ValidateContent(kVideoCodecAV1, false, 3);
}

TEST_F(IvfFileReaderTest, MultilayerH264FileNtpTimestamp) {
  CreateTestFile(kVideoCodecH264, false, 3);
  ValidateContent(kVideoCodecH264, false, 3);
}

}  // namespace webrtc
