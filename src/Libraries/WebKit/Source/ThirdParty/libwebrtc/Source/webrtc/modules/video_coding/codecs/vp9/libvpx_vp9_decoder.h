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
#ifndef MODULES_VIDEO_CODING_CODECS_VP9_LIBVPX_VP9_DECODER_H_
#define MODULES_VIDEO_CODING_CODECS_VP9_LIBVPX_VP9_DECODER_H_

#ifdef RTC_ENABLE_VP9

#include "api/video_codecs/video_decoder.h"
#include "modules/video_coding/codecs/vp9/include/vp9.h"
#include "modules/video_coding/codecs/vp9/vp9_frame_buffer_pool.h"
#include "vpx/vp8cx.h"

namespace webrtc {

class LibvpxVp9Decoder : public VP9Decoder {
 public:
  LibvpxVp9Decoder();
  virtual ~LibvpxVp9Decoder();

  bool Configure(const Settings& settings) override;

  int Decode(const EncodedImage& input_image,
             int64_t /*render_time_ms*/) override;

  int RegisterDecodeCompleteCallback(DecodedImageCallback* callback) override;

  int Release() override;

  DecoderInfo GetDecoderInfo() const override;
  const char* ImplementationName() const override;

 private:
  int ReturnFrame(const vpx_image_t* img,
                  uint32_t timestamp,
                  int qp,
                  const webrtc::ColorSpace* explicit_color_space);

  // Memory pool used to share buffers between libvpx and webrtc.
  Vp9FrameBufferPool libvpx_buffer_pool_;
  DecodedImageCallback* decode_complete_callback_;
  bool inited_;
  vpx_codec_ctx_t* decoder_;
  bool key_frame_required_;
  Settings current_settings_;
};
}  // namespace webrtc

#endif  // RTC_ENABLE_VP9

#endif  // MODULES_VIDEO_CODING_CODECS_VP9_LIBVPX_VP9_DECODER_H_
