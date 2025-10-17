/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#ifndef MODULES_VIDEO_CODING_RTP_VP8_REF_FINDER_H_
#define MODULES_VIDEO_CODING_RTP_VP8_REF_FINDER_H_

#include <deque>
#include <map>
#include <memory>
#include <set>

#include "absl/container/inlined_vector.h"
#include "modules/rtp_rtcp/source/frame_object.h"
#include "modules/video_coding/rtp_frame_reference_finder.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"

namespace webrtc {

class RtpVp8RefFinder {
 public:
  RtpVp8RefFinder() = default;

  RtpFrameReferenceFinder::ReturnVector ManageFrame(
      std::unique_ptr<RtpFrameObject> frame);
  void ClearTo(uint16_t seq_num);

 private:
  static constexpr int kFrameIdLength = 1 << 15;
  static constexpr int kMaxLayerInfo = 50;
  static constexpr int kMaxNotYetReceivedFrames = 100;
  static constexpr int kMaxStashedFrames = 100;
  static constexpr int kMaxTemporalLayers = 5;

  struct UnwrappedTl0Frame {
    int64_t unwrapped_tl0;
    std::unique_ptr<RtpFrameObject> frame;
  };

  enum FrameDecision { kStash, kHandOff, kDrop };

  FrameDecision ManageFrameInternal(RtpFrameObject* frame,
                                    const RTPVideoHeaderVP8& codec_header,
                                    int64_t unwrapped_tl0);
  void RetryStashedFrames(RtpFrameReferenceFinder::ReturnVector& res);
  void UpdateLayerInfoVp8(RtpFrameObject* frame,
                          int64_t unwrapped_tl0,
                          uint8_t temporal_idx);
  void UnwrapPictureIds(RtpFrameObject* frame);

  // Save the last picture id in order to detect when there is a gap in frames
  // that have not yet been fully received.
  int last_picture_id_ = -1;

  // Frames earlier than the last received frame that have not yet been
  // fully received.
  std::set<uint16_t, DescendingSeqNumComp<uint16_t, kFrameIdLength>>
      not_yet_received_frames_;

  // Frames that have been fully received but didn't have all the information
  // needed to determine their references.
  std::deque<UnwrappedTl0Frame> stashed_frames_;

  // Holds the information about the last completed frame for a given temporal
  // layer given an unwrapped Tl0 picture index.
  std::map<int64_t, std::array<int64_t, kMaxTemporalLayers>> layer_info_;

  // Unwrapper used to unwrap VP8/VP9 streams which have their picture id
  // specified.
  SeqNumUnwrapper<uint16_t, kFrameIdLength> unwrapper_;

  SeqNumUnwrapper<uint8_t> tl0_unwrapper_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_RTP_VP8_REF_FINDER_H_
