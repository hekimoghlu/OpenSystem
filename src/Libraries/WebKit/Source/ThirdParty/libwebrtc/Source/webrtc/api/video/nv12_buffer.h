/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 2, 2023.
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
#ifndef API_VIDEO_NV12_BUFFER_H_
#define API_VIDEO_NV12_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "api/scoped_refptr.h"
#include "api/video/video_frame_buffer.h"
#include "rtc_base/memory/aligned_malloc.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// NV12 is a biplanar encoding format, with full-resolution Y and
// half-resolution interleved UV. More information can be found at
// http://msdn.microsoft.com/library/windows/desktop/dd206750.aspx#nv12.
class RTC_EXPORT NV12Buffer : public NV12BufferInterface {
 public:
  static rtc::scoped_refptr<NV12Buffer> Create(int width, int height);
  static rtc::scoped_refptr<NV12Buffer> Create(int width,
                                               int height,
                                               int stride_y,
                                               int stride_uv);
  static rtc::scoped_refptr<NV12Buffer> Copy(
      const I420BufferInterface& i420_buffer);

  rtc::scoped_refptr<I420BufferInterface> ToI420() override;

  int width() const override;
  int height() const override;

  int StrideY() const override;
  int StrideUV() const override;

  const uint8_t* DataY() const override;
  const uint8_t* DataUV() const override;

  uint8_t* MutableDataY();
  uint8_t* MutableDataUV();

  // Sets all three planes to all zeros. Used to work around for
  // quirks in memory checkers
  // (https://bugs.chromium.org/p/libyuv/issues/detail?id=377) and
  // ffmpeg (http://crbug.com/390941).
  // TODO(https://crbug.com/390941): Deprecated. Should be deleted if/when those
  // issues are resolved in a better way. Or in the mean time, use SetBlack.
  void InitializeData();

  // Scale the cropped area of `src` to the size of `this` buffer, and
  // write the result into `this`.
  void CropAndScaleFrom(const NV12BufferInterface& src,
                        int offset_x,
                        int offset_y,
                        int crop_width,
                        int crop_height);

 protected:
  NV12Buffer(int width, int height);
  NV12Buffer(int width, int height, int stride_y, int stride_uv);

  ~NV12Buffer() override;

 private:
  size_t UVOffset() const;

  const int width_;
  const int height_;
  const int stride_y_;
  const int stride_uv_;
  const std::unique_ptr<uint8_t, AlignedFreeDeleter> data_;
};

}  // namespace webrtc

#endif  // API_VIDEO_NV12_BUFFER_H_
