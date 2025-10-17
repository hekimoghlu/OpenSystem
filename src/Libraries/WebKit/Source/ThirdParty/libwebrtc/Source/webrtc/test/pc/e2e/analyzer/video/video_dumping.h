/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_VIDEO_DUMPING_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_VIDEO_DUMPING_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "api/test/video/video_frame_writer.h"
#include "api/video/video_frame.h"
#include "api/video/video_sink_interface.h"
#include "test/testsupport/video_frame_writer.h"

namespace webrtc {
namespace webrtc_pc_e2e {

// `VideoSinkInterface` to dump incoming video frames into specified video
// writer.
class VideoWriter final : public rtc::VideoSinkInterface<VideoFrame> {
 public:
  // Creates video writer. Caller keeps ownership of `video_writer` and is
  // responsible for closing it after VideoWriter will be destroyed.
  VideoWriter(test::VideoFrameWriter* video_writer, int sampling_modulo);
  VideoWriter(const VideoWriter&) = delete;
  VideoWriter& operator=(const VideoWriter&) = delete;
  ~VideoWriter() override = default;

  void OnFrame(const VideoFrame& frame) override;

 private:
  test::VideoFrameWriter* const video_writer_;
  const int sampling_modulo_;

  int64_t frames_counter_ = 0;
};

// Creates a `VideoFrameWriter` to dump video frames together with their ids.
// It uses provided `video_writer_delegate` to write video itself. Frame ids
// will be logged into the specified file.
std::unique_ptr<test::VideoFrameWriter> CreateVideoFrameWithIdsWriter(
    std::unique_ptr<test::VideoFrameWriter> video_writer_delegate,
    absl::string_view frame_ids_dump_file_name);

}  // namespace webrtc_pc_e2e
}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_VIDEO_DUMPING_H_
