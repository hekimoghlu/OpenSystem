/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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
#ifndef MODULES_VIDEO_CODING_RTP_SEQ_NUM_ONLY_REF_FINDER_H_
#define MODULES_VIDEO_CODING_RTP_SEQ_NUM_ONLY_REF_FINDER_H_

#include <deque>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "modules/rtp_rtcp/source/frame_object.h"
#include "modules/video_coding/rtp_frame_reference_finder.h"
#include "rtc_base/numerics/sequence_number_unwrapper.h"

namespace webrtc {

class RtpSeqNumOnlyRefFinder {
 public:
  RtpSeqNumOnlyRefFinder() = default;

  RtpFrameReferenceFinder::ReturnVector ManageFrame(
      std::unique_ptr<RtpFrameObject> frame);
  RtpFrameReferenceFinder::ReturnVector PaddingReceived(uint16_t seq_num);
  void ClearTo(uint16_t seq_num);

 private:
  static constexpr int kMaxStashedFrames = 100;
  static constexpr int kMaxPaddingAge = 100;

  enum FrameDecision { kStash, kHandOff, kDrop };

  FrameDecision ManageFrameInternal(RtpFrameObject* frame);
  void RetryStashedFrames(RtpFrameReferenceFinder::ReturnVector& res);
  void UpdateLastPictureIdWithPadding(uint16_t seq_num);

  // For every group of pictures, hold two sequence numbers. The first being
  // the sequence number of the last packet of the last completed frame, and
  // the second being the sequence number of the last packet of the last
  // completed frame advanced by any potential continuous packets of padding.
  std::map<uint16_t,
           std::pair<uint16_t, uint16_t>,
           DescendingSeqNumComp<uint16_t>>
      last_seq_num_gop_;

  // Padding packets that have been received but that are not yet continuous
  // with any group of pictures.
  std::set<uint16_t, DescendingSeqNumComp<uint16_t>> stashed_padding_;

  // Frames that have been fully received but didn't have all the information
  // needed to determine their references.
  std::deque<std::unique_ptr<RtpFrameObject>> stashed_frames_;

  // Unwrapper used to unwrap generic RTP streams. In a generic stream we derive
  // a picture id from the packet sequence number.
  SeqNumUnwrapper<uint16_t> rtp_seq_num_unwrapper_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_RTP_SEQ_NUM_ONLY_REF_FINDER_H_
