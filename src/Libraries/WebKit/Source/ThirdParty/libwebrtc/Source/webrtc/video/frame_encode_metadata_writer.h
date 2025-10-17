/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#ifndef VIDEO_FRAME_ENCODE_METADATA_WRITER_H_
#define VIDEO_FRAME_ENCODE_METADATA_WRITER_H_

#include <list>
#include <optional>
#include <vector>

#include "api/video/encoded_image.h"
#include "api/video_codecs/video_codec.h"
#include "api/video_codecs/video_encoder.h"
#include "modules/video_coding/include/video_codec_interface.h"
#include "rtc_base/synchronization/mutex.h"

namespace webrtc {

class FrameEncodeMetadataWriter {
 public:
  explicit FrameEncodeMetadataWriter(EncodedImageCallback* frame_drop_callback);
  ~FrameEncodeMetadataWriter();

  void OnEncoderInit(const VideoCodec& codec);
  void OnSetRates(const VideoBitrateAllocation& bitrate_allocation,
                  uint32_t framerate_fps);

  void OnEncodeStarted(const VideoFrame& frame);

  void FillMetadataAndTimingInfo(size_t simulcast_svc_idx,
                                 EncodedImage* encoded_image);

  void UpdateBitstream(const CodecSpecificInfo* codec_specific_info,
                       EncodedImage* encoded_image);

  void Reset();

 private:
  // For non-internal-source encoders, returns encode started time and fixes
  // capture timestamp for the frame, if corrupted by the encoder.
  std::optional<int64_t> ExtractEncodeStartTimeAndFillMetadata(
      size_t simulcast_svc_idx,
      EncodedImage* encoded_image) RTC_EXCLUSIVE_LOCKS_REQUIRED(lock_);

  struct FrameMetadata {
    uint32_t rtp_timestamp;
    int64_t encode_start_time_ms;
    int64_t ntp_time_ms = 0;
    int64_t timestamp_us = 0;
    VideoRotation rotation = kVideoRotation_0;
    std::optional<ColorSpace> color_space;
    bool is_steady_state_refresh_frame = false;
    RtpPacketInfos packet_infos;
  };
  struct TimingFramesLayerInfo {
    TimingFramesLayerInfo();
    ~TimingFramesLayerInfo();
    size_t target_bitrate_bytes_per_sec = 0;
    std::list<FrameMetadata> frames;
  };

  Mutex lock_;
  EncodedImageCallback* const frame_drop_callback_;
  VideoCodec codec_settings_ RTC_GUARDED_BY(&lock_);
  uint32_t framerate_fps_ RTC_GUARDED_BY(&lock_);

  size_t num_spatial_layers_ RTC_GUARDED_BY(&lock_);
  // Separate instance for each simulcast stream or spatial layer.
  std::vector<TimingFramesLayerInfo> timing_frames_info_ RTC_GUARDED_BY(&lock_);
  int64_t last_timing_frame_time_ms_ RTC_GUARDED_BY(&lock_);
  size_t reordered_frames_logged_messages_ RTC_GUARDED_BY(&lock_);
  size_t stalled_encoder_logged_messages_ RTC_GUARDED_BY(&lock_);
};

}  // namespace webrtc

#endif  // VIDEO_FRAME_ENCODE_METADATA_WRITER_H_
