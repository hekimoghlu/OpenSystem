/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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
#include "modules/video_coding/utility/ivf_file_writer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "api/video/encoded_image.h"
#include "api/video/video_codec_type.h"
#include "api/video_codecs/video_codec.h"
#include "modules/rtp_rtcp/source/byte_io.h"
#include "modules/video_coding/utility/ivf_defines.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"
#include "rtc_base/system/file_wrapper.h"

// TODO(palmkvist): make logging more informative in the absence of a file name
// (or get one)

namespace webrtc {

namespace {

constexpr int kDefaultWidth = 1280;
constexpr int kDefaultHeight = 720;
}  // namespace

IvfFileWriter::IvfFileWriter(FileWrapper file, size_t byte_limit)
    : codec_type_(kVideoCodecGeneric),
      bytes_written_(0),
      byte_limit_(byte_limit),
      num_frames_(0),
      width_(0),
      height_(0),
      last_timestamp_(-1),
      using_capture_timestamps_(false),
      file_(std::move(file)) {
  RTC_DCHECK(byte_limit == 0 || kIvfHeaderSize <= byte_limit)
      << "The byte_limit is too low, not even the header will fit.";
}

IvfFileWriter::~IvfFileWriter() {
  Close();
}

std::unique_ptr<IvfFileWriter> IvfFileWriter::Wrap(FileWrapper file,
                                                   size_t byte_limit) {
  return std::unique_ptr<IvfFileWriter>(
      new IvfFileWriter(std::move(file), byte_limit));
}

std::unique_ptr<IvfFileWriter> IvfFileWriter::Wrap(absl::string_view filename,
                                                   size_t byte_limit) {
  return std::unique_ptr<IvfFileWriter>(
      new IvfFileWriter(FileWrapper::OpenWriteOnly(filename), byte_limit));
}

bool IvfFileWriter::WriteHeader() {
  if (!file_.Rewind()) {
    RTC_LOG(LS_WARNING) << "Unable to rewind ivf output file.";
    return false;
  }

  uint8_t ivf_header[kIvfHeaderSize] = {0};
  ivf_header[0] = 'D';
  ivf_header[1] = 'K';
  ivf_header[2] = 'I';
  ivf_header[3] = 'F';
  ByteWriter<uint16_t>::WriteLittleEndian(&ivf_header[4], 0);   // Version.
  ByteWriter<uint16_t>::WriteLittleEndian(&ivf_header[6], 32);  // Header size.

  switch (codec_type_) {
    case kVideoCodecVP8:
      ivf_header[8] = 'V';
      ivf_header[9] = 'P';
      ivf_header[10] = '8';
      ivf_header[11] = '0';
      break;
    case kVideoCodecVP9:
      ivf_header[8] = 'V';
      ivf_header[9] = 'P';
      ivf_header[10] = '9';
      ivf_header[11] = '0';
      break;
    case kVideoCodecAV1:
      ivf_header[8] = 'A';
      ivf_header[9] = 'V';
      ivf_header[10] = '0';
      ivf_header[11] = '1';
      break;
    case kVideoCodecH264:
      ivf_header[8] = 'H';
      ivf_header[9] = '2';
      ivf_header[10] = '6';
      ivf_header[11] = '4';
      break;
    case kVideoCodecH265:
      ivf_header[8] = 'H';
      ivf_header[9] = '2';
      ivf_header[10] = '6';
      ivf_header[11] = '5';
      break;
    default:
      // For unknown codec type use **** code. You can specify actual payload
      // format when playing the video with ffplay: ffplay -f H263 file.ivf
      ivf_header[8] = '*';
      ivf_header[9] = '*';
      ivf_header[10] = '*';
      ivf_header[11] = '*';
      break;
  }

  ByteWriter<uint16_t>::WriteLittleEndian(&ivf_header[12], width_);
  ByteWriter<uint16_t>::WriteLittleEndian(&ivf_header[14], height_);
  // Render timestamps are in ms (1/1000 scale), while RTP timestamps use a
  // 90kHz clock.
  ByteWriter<uint32_t>::WriteLittleEndian(
      &ivf_header[16], using_capture_timestamps_ ? 1000 : 90000);
  ByteWriter<uint32_t>::WriteLittleEndian(&ivf_header[20], 1);
  ByteWriter<uint32_t>::WriteLittleEndian(&ivf_header[24],
                                          static_cast<uint32_t>(num_frames_));
  ByteWriter<uint32_t>::WriteLittleEndian(&ivf_header[28], 0);  // Reserved.

  if (!file_.Write(ivf_header, kIvfHeaderSize)) {
    RTC_LOG(LS_ERROR) << "Unable to write IVF header for ivf output file.";
    return false;
  }

  if (bytes_written_ < kIvfHeaderSize) {
    bytes_written_ = kIvfHeaderSize;
  }

  return true;
}

bool IvfFileWriter::InitFromFirstFrame(const EncodedImage& encoded_image,
                                       VideoCodecType codec_type) {
  if (encoded_image._encodedWidth == 0 || encoded_image._encodedHeight == 0) {
    width_ = kDefaultWidth;
    height_ = kDefaultHeight;
  } else {
    width_ = encoded_image._encodedWidth;
    height_ = encoded_image._encodedHeight;
  }

  using_capture_timestamps_ = encoded_image.RtpTimestamp() == 0;

  codec_type_ = codec_type;

  if (!WriteHeader())
    return false;

  const char* codec_name = CodecTypeToPayloadString(codec_type_);
  RTC_LOG(LS_WARNING) << "Created IVF file for codec data of type "
                      << codec_name << " at resolution " << width_ << " x "
                      << height_ << ", using "
                      << (using_capture_timestamps_ ? "1" : "90")
                      << "kHz clock resolution.";
  return true;
}

bool IvfFileWriter::WriteFrame(const EncodedImage& encoded_image,
                               VideoCodecType codec_type) {
  if (!file_.is_open())
    return false;

  if (num_frames_ == 0 && !InitFromFirstFrame(encoded_image, codec_type))
    return false;
  RTC_DCHECK_EQ(codec_type_, codec_type);

  int64_t timestamp = using_capture_timestamps_
                          ? encoded_image.capture_time_ms_
                          : wrap_handler_.Unwrap(encoded_image.RtpTimestamp());
  if (last_timestamp_ != -1 && timestamp < last_timestamp_) {
    RTC_LOG(LS_WARNING) << "Timestamp not increasing: " << last_timestamp_
                        << " -> " << timestamp;
  }
  last_timestamp_ = timestamp;

  bool written_frames = false;
  size_t max_sl_index = encoded_image.SpatialIndex().value_or(0);
  const uint8_t* data = encoded_image.data();
  for (size_t sl_idx = 0; sl_idx <= max_sl_index; ++sl_idx) {
    size_t cur_size = encoded_image.SpatialLayerFrameSize(sl_idx).value_or(0);
    if (cur_size > 0) {
      written_frames = true;
      if (!WriteOneSpatialLayer(timestamp, data, cur_size)) {
        return false;
      }
      data += cur_size;
    }
  }

  // If frame has only one spatial layer it won't have any spatial layers'
  // sizes. Therefore this case should be addressed separately.
  if (!written_frames) {
    return WriteOneSpatialLayer(timestamp, data, encoded_image.size());
  } else {
    return true;
  }
}

bool IvfFileWriter::WriteOneSpatialLayer(int64_t timestamp,
                                         const uint8_t* data,
                                         size_t size) {
  const size_t kFrameHeaderSize = 12;
  if (byte_limit_ != 0 &&
      bytes_written_ + kFrameHeaderSize + size > byte_limit_) {
    RTC_LOG(LS_WARNING) << "Closing IVF file due to reaching size limit: "
                        << byte_limit_ << " bytes.";
    Close();
    return false;
  }
  uint8_t frame_header[kFrameHeaderSize] = {};
  ByteWriter<uint32_t>::WriteLittleEndian(&frame_header[0],
                                          static_cast<uint32_t>(size));
  ByteWriter<uint64_t>::WriteLittleEndian(&frame_header[4], timestamp);
  if (!file_.Write(frame_header, kFrameHeaderSize) ||
      !file_.Write(data, size)) {
    RTC_LOG(LS_ERROR) << "Unable to write frame to file.";
    return false;
  }

  bytes_written_ += kFrameHeaderSize + size;

  ++num_frames_;
  return true;
}

bool IvfFileWriter::Close() {
  if (!file_.is_open())
    return false;

  if (num_frames_ == 0) {
    file_.Close();
    return true;
  }

  bool ret = WriteHeader();
  file_.Close();
  return ret;
}

}  // namespace webrtc
