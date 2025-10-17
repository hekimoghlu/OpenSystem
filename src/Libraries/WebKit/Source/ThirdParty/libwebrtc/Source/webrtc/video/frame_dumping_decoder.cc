/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#include "video/frame_dumping_decoder.h"

#include <memory>
#include <utility>

#include "modules/video_coding/include/video_codec_interface.h"
#include "modules/video_coding/utility/ivf_file_writer.h"

namespace webrtc {
namespace {

class FrameDumpingDecoder : public VideoDecoder {
 public:
  FrameDumpingDecoder(std::unique_ptr<VideoDecoder> decoder, FileWrapper file);
  ~FrameDumpingDecoder() override;

  bool Configure(const Settings& settings) override;
  int32_t Decode(const EncodedImage& input_image,
                 int64_t render_time_ms) override;
  int32_t RegisterDecodeCompleteCallback(
      DecodedImageCallback* callback) override;
  int32_t Release() override;
  DecoderInfo GetDecoderInfo() const override;
  const char* ImplementationName() const override;

 private:
  std::unique_ptr<VideoDecoder> decoder_;
  VideoCodecType codec_type_ = VideoCodecType::kVideoCodecGeneric;
  std::unique_ptr<IvfFileWriter> writer_;
};

FrameDumpingDecoder::FrameDumpingDecoder(std::unique_ptr<VideoDecoder> decoder,
                                         FileWrapper file)
    : decoder_(std::move(decoder)),
      writer_(IvfFileWriter::Wrap(std::move(file),
                                  /* byte_limit= */ 100000000)) {}

FrameDumpingDecoder::~FrameDumpingDecoder() = default;

bool FrameDumpingDecoder::Configure(const Settings& settings) {
  codec_type_ = settings.codec_type();
  return decoder_->Configure(settings);
}

int32_t FrameDumpingDecoder::Decode(const EncodedImage& input_image,
                                    int64_t render_time_ms) {
  int32_t ret = decoder_->Decode(input_image, render_time_ms);
  writer_->WriteFrame(input_image, codec_type_);

  return ret;
}

int32_t FrameDumpingDecoder::RegisterDecodeCompleteCallback(
    DecodedImageCallback* callback) {
  return decoder_->RegisterDecodeCompleteCallback(callback);
}

int32_t FrameDumpingDecoder::Release() {
  return decoder_->Release();
}

VideoDecoder::DecoderInfo FrameDumpingDecoder::GetDecoderInfo() const {
  return decoder_->GetDecoderInfo();
}

const char* FrameDumpingDecoder::ImplementationName() const {
  return decoder_->ImplementationName();
}

}  // namespace

std::unique_ptr<VideoDecoder> CreateFrameDumpingDecoderWrapper(
    std::unique_ptr<VideoDecoder> decoder,
    FileWrapper file) {
  return std::make_unique<FrameDumpingDecoder>(std::move(decoder),
                                               std::move(file));
}

}  // namespace webrtc
