/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#ifndef MODULES_VIDEO_CODING_UTILITY_IVF_FILE_READER_H_
#define MODULES_VIDEO_CODING_UTILITY_IVF_FILE_READER_H_

#include <memory>
#include <optional>
#include <utility>

#include "api/video/encoded_image.h"
#include "api/video_codecs/video_codec.h"
#include "rtc_base/system/file_wrapper.h"

namespace webrtc {

class IvfFileReader {
 public:
  // Creates IvfFileReader. Returns nullptr if error acquired.
  static std::unique_ptr<IvfFileReader> Create(FileWrapper file);
  ~IvfFileReader();

  IvfFileReader(const IvfFileReader&) = delete;
  IvfFileReader& operator=(const IvfFileReader&) = delete;

  // Reinitializes reader. Returns false if any error acquired.
  bool Reset();

  // Returns codec type which was used to create this IVF file and which should
  // be used to decode EncodedImages from this file.
  VideoCodecType GetVideoCodecType() const { return codec_type_; }
  // Returns count of frames in this file.
  size_t GetFramesCount() const { return num_frames_; }

  // Returns next frame or std::nullopt if any error acquired. Always returns
  // std::nullopt after first error was spotted.
  std::optional<EncodedImage> NextFrame();
  bool HasMoreFrames() const { return num_read_frames_ < num_frames_; }
  bool HasError() const { return has_error_; }

  uint16_t GetFrameWidth() const { return width_; }
  uint16_t GetFrameHeight() const { return height_; }

  bool Close();

 private:
  struct FrameHeader {
    size_t frame_size;
    int64_t timestamp;
  };

  explicit IvfFileReader(FileWrapper file) : file_(std::move(file)) {}

  // Parses codec type from specified position of the buffer. Codec type
  // contains kCodecTypeBytesCount bytes and caller has to ensure that buffer
  // won't overflow.
  std::optional<VideoCodecType> ParseCodecType(uint8_t* buffer,
                                               size_t start_pos);
  std::optional<FrameHeader> ReadNextFrameHeader();

  VideoCodecType codec_type_;
  size_t num_frames_;
  size_t num_read_frames_;
  uint16_t width_;
  uint16_t height_;
  uint32_t time_scale_;
  FileWrapper file_;

  std::optional<FrameHeader> next_frame_header_;
  bool has_error_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_UTILITY_IVF_FILE_READER_H_
