/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#ifndef RTC_TOOLS_VIDEO_ENCODER_ENCODED_IMAGE_FILE_WRITER_H_
#define RTC_TOOLS_VIDEO_ENCODER_ENCODED_IMAGE_FILE_WRITER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "modules/video_coding/include/video_codec_interface.h"
#include "modules/video_coding/utility/ivf_file_writer.h"

namespace webrtc {
namespace test {

// The `EncodedImageFileWriter` writes the `EncodedImage` into ivf output. It
// supports SVC to output ivf for all decode targets.
class EncodedImageFileWriter final {
  // The pair of writer and output file name.
  using IvfWriterPair = std::pair<std::unique_ptr<IvfFileWriter>, std::string>;

 public:
  explicit EncodedImageFileWriter(const VideoCodec& video_codec_setting);

  ~EncodedImageFileWriter();

  int Write(const EncodedImage& encoded_image);

 private:
  VideoCodec video_codec_setting_;

  int spatial_layers_ = 0;
  int temporal_layers_ = 0;
  InterLayerPredMode inter_layer_pred_mode_ = InterLayerPredMode::kOff;

  bool is_base_layer_key_frame = false;
  std::vector<IvfWriterPair> decode_target_writers_;
};

}  // namespace test
}  // namespace webrtc

#endif  // RTC_TOOLS_VIDEO_ENCODER_ENCODED_IMAGE_FILE_WRITER_H_
