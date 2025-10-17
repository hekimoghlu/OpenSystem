/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#include <stdio.h>

#include <string>

#include "api/scoped_refptr.h"
#include "api/video/i420_buffer.h"
#include "common_video/libyuv/include/webrtc_libyuv.h"
#include "rtc_base/logging.h"
#include "test/frame_utils.h"
#include "test/testsupport/file_utils.h"
#include "test/testsupport/frame_reader.h"

namespace webrtc {
namespace test {
namespace {
using RepeatMode = YuvFrameReaderImpl::RepeatMode;

int WrapFrameNum(int frame_num, int num_frames, RepeatMode mode) {
  RTC_CHECK_GE(frame_num, 0) << "frame_num cannot be negative";
  RTC_CHECK_GT(num_frames, 0) << "num_frames must be greater than 0";
  if (mode == RepeatMode::kSingle) {
    return frame_num;
  }
  if (mode == RepeatMode::kRepeat) {
    return frame_num % num_frames;
  }

  RTC_CHECK_EQ(RepeatMode::kPingPong, mode);
  int cycle_len = std::max(1, 2 * (num_frames - 1));
  int wrapped_num = frame_num % cycle_len;
  if (wrapped_num >= num_frames) {
    return cycle_len - wrapped_num;
  }
  return wrapped_num;
}

rtc::scoped_refptr<I420Buffer> Scale(rtc::scoped_refptr<I420Buffer> buffer,
                                     Resolution resolution) {
  if (buffer->width() == resolution.width &&
      buffer->height() == resolution.height) {
    return buffer;
  }
  rtc::scoped_refptr<I420Buffer> scaled(
      I420Buffer::Create(resolution.width, resolution.height));
  scaled->ScaleFrom(*buffer.get());
  return scaled;
}
}  // namespace

int YuvFrameReaderImpl::RateScaler::Skip(Ratio framerate_scale) {
  ticks_ = ticks_.value_or(framerate_scale.num);
  int skip = 0;
  while (ticks_ <= 0) {
    *ticks_ += framerate_scale.num;
    ++skip;
  }
  *ticks_ -= framerate_scale.den;
  return skip;
}

YuvFrameReaderImpl::YuvFrameReaderImpl(std::string filepath,
                                       Resolution resolution,
                                       RepeatMode repeat_mode)
    : filepath_(filepath),
      resolution_(resolution),
      repeat_mode_(repeat_mode),
      num_frames_(0),
      frame_num_(0),
      frame_size_bytes_(0),
      header_size_bytes_(0),
      file_(nullptr) {}

YuvFrameReaderImpl::~YuvFrameReaderImpl() {
  if (file_ != nullptr) {
    fclose(file_);
    file_ = nullptr;
  }
}

void YuvFrameReaderImpl::Init() {
  RTC_CHECK_GT(resolution_.width, 0) << "Width must be positive";
  RTC_CHECK_GT(resolution_.height, 0) << "Height must be positive";
  frame_size_bytes_ =
      CalcBufferSize(VideoType::kI420, resolution_.width, resolution_.height);

  file_ = fopen(filepath_.c_str(), "rb");
  RTC_CHECK(file_ != NULL) << "Cannot open " << filepath_;

  size_t file_size_bytes = GetFileSize(filepath_);
  RTC_CHECK_GT(file_size_bytes, 0u) << "File " << filepath_ << " is empty";

  num_frames_ = static_cast<int>(file_size_bytes / frame_size_bytes_);
  RTC_CHECK_GT(num_frames_, 0u) << "File " << filepath_ << " is too small";
}

rtc::scoped_refptr<I420Buffer> YuvFrameReaderImpl::PullFrame() {
  return PullFrame(/*frame_num=*/nullptr);
}

rtc::scoped_refptr<I420Buffer> YuvFrameReaderImpl::PullFrame(int* frame_num) {
  return PullFrame(frame_num, resolution_, /*framerate_scale=*/kNoScale);
}

rtc::scoped_refptr<I420Buffer> YuvFrameReaderImpl::PullFrame(
    int* frame_num,
    Resolution resolution,
    Ratio framerate_scale) {
  frame_num_ += framerate_scaler_.Skip(framerate_scale);
  auto buffer = ReadFrame(frame_num_, resolution);
  if (frame_num != nullptr) {
    *frame_num = frame_num_;
  }
  return buffer;
}

rtc::scoped_refptr<I420Buffer> YuvFrameReaderImpl::ReadFrame(int frame_num) {
  return ReadFrame(frame_num, resolution_);
}

rtc::scoped_refptr<I420Buffer> YuvFrameReaderImpl::ReadFrame(
    int frame_num,
    Resolution resolution) {
  int wrapped_num = WrapFrameNum(frame_num, num_frames_, repeat_mode_);
  if (wrapped_num >= num_frames_) {
    RTC_CHECK_EQ(RepeatMode::kSingle, repeat_mode_);
    return nullptr;
  }
  fseek(file_, header_size_bytes_ + wrapped_num * frame_size_bytes_, SEEK_SET);
  auto buffer = ReadI420Buffer(resolution_.width, resolution_.height, file_);
  RTC_CHECK(buffer != nullptr);

  return Scale(buffer, resolution);
}

std::unique_ptr<FrameReader> CreateYuvFrameReader(std::string filepath,
                                                  Resolution resolution) {
  return CreateYuvFrameReader(filepath, resolution,
                              YuvFrameReaderImpl::RepeatMode::kSingle);
}

std::unique_ptr<FrameReader> CreateYuvFrameReader(
    std::string filepath,
    Resolution resolution,
    YuvFrameReaderImpl::RepeatMode repeat_mode) {
  YuvFrameReaderImpl* frame_reader =
      new YuvFrameReaderImpl(filepath, resolution, repeat_mode);
  frame_reader->Init();
  return std::unique_ptr<FrameReader>(frame_reader);
}

}  // namespace test
}  // namespace webrtc
