/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
#ifndef VIDEO_VIDEO_STREAM_DECODER2_H_
#define VIDEO_VIDEO_STREAM_DECODER2_H_

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "api/scoped_refptr.h"
#include "api/units/time_delta.h"
#include "api/video/video_content_type.h"
#include "api/video/video_frame.h"
#include "api/video/video_frame_type.h"
#include "api/video/video_sink_interface.h"
#include "api/video_codecs/video_decoder.h"
#include "modules/remote_bitrate_estimator/include/remote_bitrate_estimator.h"
#include "modules/video_coding/include/video_coding_defines.h"
#include "rtc_base/platform_thread.h"

namespace webrtc {

class VideoReceiver2;

namespace internal {

class ReceiveStatisticsProxy;

class VideoStreamDecoder : public VCMReceiveCallback {
 public:
  VideoStreamDecoder(
      VideoReceiver2* video_receiver,
      ReceiveStatisticsProxy* receive_statistics_proxy,
      rtc::VideoSinkInterface<VideoFrame>* incoming_video_stream);
  ~VideoStreamDecoder() override;

  // Implements VCMReceiveCallback.
  int32_t FrameToRender(VideoFrame& video_frame,
                        std::optional<uint8_t> qp,
                        TimeDelta decode_time,
                        VideoContentType content_type,
                        VideoFrameType frame_type) override;
  int32_t OnFrameToRender(const struct FrameToRender& arguments) override;
  void OnDroppedFrames(uint32_t frames_dropped) override;
  void OnIncomingPayloadType(int payload_type) override;
  void OnDecoderInfoChanged(
      const VideoDecoder::DecoderInfo& decoder_info) override;

 private:
  VideoReceiver2* const video_receiver_;
  ReceiveStatisticsProxy* const receive_stats_callback_;
  rtc::VideoSinkInterface<VideoFrame>* const incoming_video_stream_;
};

}  // namespace internal
}  // namespace webrtc

#endif  // VIDEO_VIDEO_STREAM_DECODER2_H_
