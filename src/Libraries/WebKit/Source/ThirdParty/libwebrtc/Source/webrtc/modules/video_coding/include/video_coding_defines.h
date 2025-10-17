/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#ifndef MODULES_VIDEO_CODING_INCLUDE_VIDEO_CODING_DEFINES_H_
#define MODULES_VIDEO_CODING_INCLUDE_VIDEO_CODING_DEFINES_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "api/units/time_delta.h"
#include "api/video/video_content_type.h"
#include "api/video/video_frame.h"
#include "api/video/video_frame_type.h"
#include "api/video/video_timing.h"
#include "api/video_codecs/video_decoder.h"
#include "rtc_base/checks.h"

namespace webrtc {

// Error codes
#define VCM_FRAME_NOT_READY 3
#define VCM_MISSING_CALLBACK 1
#define VCM_OK 0
#define VCM_GENERAL_ERROR -1
#define VCM_PARAMETER_ERROR -4
#define VCM_NO_CODEC_REGISTERED -8
#define VCM_JITTER_BUFFER_ERROR -9

enum {
  // Timing frames settings. Timing frames are sent every
  // `kDefaultTimingFramesDelayMs`, or if the frame is at least
  // `kDefaultOutlierFrameSizePercent` in size of average frame.
  kDefaultTimingFramesDelayMs = 200,
  kDefaultOutlierFrameSizePercent = 500,
  // Maximum number of frames for what we store encode start timing information.
  kMaxEncodeStartTimeListSize = 150,
};

enum VCMVideoProtection {
  kProtectionNack,
  kProtectionNackFEC,
};

// Callback class used for passing decoded frames which are ready to be
// rendered.
class VCMReceiveCallback {
 public:
  struct FrameToRender {
    VideoFrame& video_frame;
    std::optional<uint8_t> qp;
    TimeDelta decode_time;
    VideoContentType content_type;
    VideoFrameType frame_type;
    std::optional<double> corruption_score;
  };

  // TODO: bugs.webrtc.org/358039777 - Delete this function.
  virtual int32_t FrameToRender(VideoFrame& videoFrame,  // NOLINT
                                std::optional<uint8_t> qp,
                                TimeDelta decode_time,
                                VideoContentType content_type,
                                VideoFrameType frame_type) {
    RTC_CHECK_NOTREACHED();
    return 0;
  }

  // TODO: bugs.webrtc.org/358039777 - Make this pure virtual.
  virtual int32_t OnFrameToRender(const struct FrameToRender& arguments) {
    return FrameToRender(arguments.video_frame, arguments.qp,
                         arguments.decode_time, arguments.content_type,
                         arguments.frame_type);
  }

  virtual void OnDroppedFrames(uint32_t frames_dropped);

  // Called when the current receive codec changes.
  virtual void OnIncomingPayloadType(int payload_type);
  virtual void OnDecoderInfoChanged(
      const VideoDecoder::DecoderInfo& decoder_info);

 protected:
  virtual ~VCMReceiveCallback() {}
};

// Callback class used for telling the user about what frame type needed to
// continue decoding.
// Typically a key frame when the stream has been corrupted in some way.
class VCMFrameTypeCallback {
 public:
  virtual int32_t RequestKeyFrame() = 0;

 protected:
  virtual ~VCMFrameTypeCallback() {}
};

// Callback class used for telling the user about which packet sequence numbers
// are currently
// missing and need to be resent.
// TODO(philipel): Deprecate VCMPacketRequestCallback
//                 and use NackSender instead.
class VCMPacketRequestCallback {
 public:
  virtual int32_t ResendPackets(const uint16_t* sequenceNumbers,
                                uint16_t length) = 0;

 protected:
  virtual ~VCMPacketRequestCallback() {}
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_INCLUDE_VIDEO_CODING_DEFINES_H_
