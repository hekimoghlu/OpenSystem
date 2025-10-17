/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
#ifndef MODULES_VIDEO_CODING_UTILITY_IVF_FILE_WRITER_H_
#define MODULES_VIDEO_CODING_UTILITY_IVF_FILE_WRITER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "absl/strings/string_view.h"
#include "api/video/encoded_image.h"
#include "api/video/video_codec_type.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"
#include "rtc_base/system/file_wrapper.h"

namespace webrtc {

class IvfFileWriter {
 public:
  // Takes ownership of the file, which will be closed either through
  // Close or ~IvfFileWriter. If writing a frame would take the file above the
  // `byte_limit` the file will be closed, the write (and all future writes)
  // will fail. A `byte_limit` of 0 is equivalent to no limit.
  static std::unique_ptr<IvfFileWriter> Wrap(FileWrapper file,
                                             size_t byte_limit);
  static std::unique_ptr<IvfFileWriter> Wrap(absl::string_view filename,
                                             size_t byte_limit);
  ~IvfFileWriter();

  IvfFileWriter(const IvfFileWriter&) = delete;
  IvfFileWriter& operator=(const IvfFileWriter&) = delete;

  bool WriteFrame(const EncodedImage& encoded_image, VideoCodecType codec_type);
  bool Close();

 private:
  explicit IvfFileWriter(FileWrapper file, size_t byte_limit);

  bool WriteHeader();
  bool InitFromFirstFrame(const EncodedImage& encoded_image,
                          VideoCodecType codec_type);
  bool WriteOneSpatialLayer(int64_t timestamp,
                            const uint8_t* data,
                            size_t size);

  VideoCodecType codec_type_;
  size_t bytes_written_;
  size_t byte_limit_;
  size_t num_frames_;
  uint16_t width_;
  uint16_t height_;
  int64_t last_timestamp_;
  bool using_capture_timestamps_;
  RtpTimestampUnwrapper wrap_handler_;
  FileWrapper file_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_UTILITY_IVF_FILE_WRITER_H_
