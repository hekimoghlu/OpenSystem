/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#ifndef MODULES_VIDEO_CODING_DEPRECATED_FRAME_BUFFER_H_
#define MODULES_VIDEO_CODING_DEPRECATED_FRAME_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "modules/video_coding/codecs/vp9/include/vp9_globals.h"
#include "modules/video_coding/deprecated/jitter_buffer_common.h"
#include "modules/video_coding/deprecated/packet.h"
#include "modules/video_coding/deprecated/session_info.h"
#include "modules/video_coding/encoded_frame.h"

namespace webrtc {

class VCMFrameBuffer : public VCMEncodedFrame {
 public:
  VCMFrameBuffer();
  virtual ~VCMFrameBuffer();

  virtual void Reset();

  VCMFrameBufferEnum InsertPacket(const VCMPacket& packet,
                                  int64_t timeInMs,
                                  const FrameData& frame_data);

  // State
  // Get current state of frame
  VCMFrameBufferStateEnum GetState() const;
  void PrepareForDecode(bool continuous);

  bool IsSessionComplete() const;
  bool HaveFirstPacket() const;
  int NumPackets() const;

  // Sequence numbers
  // Get lowest packet sequence number in frame
  int32_t GetLowSeqNum() const;
  // Get highest packet sequence number in frame
  int32_t GetHighSeqNum() const;

  int PictureId() const;
  int TemporalId() const;
  bool LayerSync() const;
  int Tl0PicId() const;

  std::vector<NaluInfo> GetNaluInfos() const;

  void SetGofInfo(const GofInfoVP9& gof_info, size_t idx);

  // Increments a counter to keep track of the number of packets of this frame
  // which were NACKed before they arrived.
  void IncrementNackCount();
  // Returns the number of packets of this frame which were NACKed before they
  // arrived.
  int16_t GetNackCount() const;

  int64_t LatestPacketTimeMs() const;

  webrtc::VideoFrameType FrameType() const;

 private:
  void SetState(VCMFrameBufferStateEnum state);  // Set state of frame

  VCMFrameBufferStateEnum _state;  // Current state of the frame
  // Set with SetEncodedData, but keep pointer to the concrete class here, to
  // enable reallocation and mutation.
  rtc::scoped_refptr<EncodedImageBuffer> encoded_image_buffer_;
  VCMSessionInfo _sessionInfo;
  uint16_t _nackCount;
  int64_t _latestPacketTimeMs;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_DEPRECATED_FRAME_BUFFER_H_
