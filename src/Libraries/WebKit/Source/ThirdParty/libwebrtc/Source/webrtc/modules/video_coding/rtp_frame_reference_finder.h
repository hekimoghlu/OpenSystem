/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#ifndef MODULES_VIDEO_CODING_RTP_FRAME_REFERENCE_FINDER_H_
#define MODULES_VIDEO_CODING_RTP_FRAME_REFERENCE_FINDER_H_

#include <memory>

#include "modules/rtp_rtcp/source/frame_object.h"

namespace webrtc {
namespace internal {
class RtpFrameReferenceFinderImpl;
}  // namespace internal

class RtpFrameReferenceFinder {
 public:
  using ReturnVector = absl::InlinedVector<std::unique_ptr<RtpFrameObject>, 3>;

  RtpFrameReferenceFinder();
  explicit RtpFrameReferenceFinder(int64_t picture_id_offset);
  ~RtpFrameReferenceFinder();

  // The RtpFrameReferenceFinder will hold onto the frame until:
  //  - the required information to determine its references has been received,
  //    in which case it (and possibly other) frames are returned, or
  //  - There are too many stashed frames (determined by `kMaxStashedFrames`),
  //    in which case it gets dropped, or
  //  - It gets cleared by ClearTo, in which case its dropped.
  //  - The frame is old, in which case it also gets dropped.
  ReturnVector ManageFrame(std::unique_ptr<RtpFrameObject> frame);

  // Notifies that padding has been received, which the reference finder
  // might need to calculate the references of a frame.
  ReturnVector PaddingReceived(uint16_t seq_num);

  // Clear all stashed frames that include packets older than `seq_num`.
  void ClearTo(uint16_t seq_num);

 private:
  void AddPictureIdOffset(ReturnVector& frames);

  // How far frames have been cleared out of the buffer by RTP sequence number.
  // A frame will be cleared if it contains a packet with a sequence number
  // older than `cleared_to_seq_num_`.
  int cleared_to_seq_num_ = -1;
  const int64_t picture_id_offset_;
  std::unique_ptr<internal::RtpFrameReferenceFinderImpl> impl_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_RTP_FRAME_REFERENCE_FINDER_H_
