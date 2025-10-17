/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#include "modules/rtp_rtcp/source/active_decode_targets_helper.h"

#include <stdint.h>

#include "api/array_view.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace {

// Returns mask of ids of chains previous frame is part of.
// Assumes for each chain frames are seen in order and no frame on any chain is
// missing. That assumptions allows a simple detection when previous frame is
// part of a chain.
std::bitset<32> LastSendOnChain(int frame_diff,
                                rtc::ArrayView<const int> chain_diffs) {
  std::bitset<32> bitmask = 0;
  for (size_t i = 0; i < chain_diffs.size(); ++i) {
    if (frame_diff == chain_diffs[i]) {
      bitmask.set(i);
    }
  }
  return bitmask;
}

// Returns bitmask with first `num` bits set to 1.
std::bitset<32> AllActive(size_t num) {
  RTC_DCHECK_LE(num, 32);
  return (~uint32_t{0}) >> (32 - num);
}

// Returns bitmask of chains that protect at least one active decode target.
std::bitset<32> ActiveChains(
    rtc::ArrayView<const int> decode_target_protected_by_chain,
    int num_chains,
    std::bitset<32> active_decode_targets) {
  std::bitset<32> active_chains = 0;
  for (size_t dt = 0; dt < decode_target_protected_by_chain.size(); ++dt) {
    if (dt < active_decode_targets.size() && !active_decode_targets[dt]) {
      continue;
    }
    int chain_idx = decode_target_protected_by_chain[dt];
    RTC_DCHECK_LT(chain_idx, num_chains);
    active_chains.set(chain_idx);
  }
  return active_chains;
}

}  // namespace

void ActiveDecodeTargetsHelper::OnFrame(
    rtc::ArrayView<const int> decode_target_protected_by_chain,
    std::bitset<32> active_decode_targets,
    bool is_keyframe,
    int64_t frame_id,
    rtc::ArrayView<const int> chain_diffs) {
  const int num_chains = chain_diffs.size();
  if (num_chains == 0) {
    // Avoid printing the warning
    // when already printed the warning for the same active decode targets, or
    // when active_decode_targets are not changed from it's default value of
    // all are active, including non-existent decode targets.
    if (last_active_decode_targets_ != active_decode_targets &&
        !active_decode_targets.all()) {
      RTC_LOG(LS_WARNING) << "No chains are configured, but some decode "
                             "targets might be inactive. Unsupported.";
    }
    last_active_decode_targets_ = active_decode_targets;
    return;
  }
  const size_t num_decode_targets = decode_target_protected_by_chain.size();
  RTC_DCHECK_GT(num_decode_targets, 0);
  std::bitset<32> all_decode_targets = AllActive(num_decode_targets);
  // Default value for active_decode_targets is 'all are active', i.e. all bits
  // are set. Default value is set before number of decode targets is known.
  // It is up to this helper to make the value cleaner and unset unused bits.
  active_decode_targets &= all_decode_targets;

  if (is_keyframe) {
    // Key frame resets the state.
    last_active_decode_targets_ = all_decode_targets;
    last_active_chains_ = AllActive(num_chains);
    unsent_on_chain_.reset();
  } else {
    // Update state assuming previous frame was sent.
    unsent_on_chain_ &=
        ~LastSendOnChain(frame_id - last_frame_id_, chain_diffs);
  }
  // Save for the next call to OnFrame.
  // Though usually `frame_id == last_frame_id_ + 1`, it might not be so when
  // frame id space is shared by several simulcast rtp streams.
  last_frame_id_ = frame_id;

  if (active_decode_targets == last_active_decode_targets_) {
    return;
  }
  last_active_decode_targets_ = active_decode_targets;

  if (active_decode_targets.none()) {
    RTC_LOG(LS_ERROR) << "It is invalid to produce a frame (" << frame_id
                      << ") while there are no active decode targets";
    return;
  }
  last_active_chains_ = ActiveChains(decode_target_protected_by_chain,
                                     num_chains, active_decode_targets);
  // Frames that are part of inactive chains might not be produced by the
  // encoder. Thus stop sending `active_decode_target` bitmask when it is sent
  // on all active chains rather than on all chains.
  unsent_on_chain_ = last_active_chains_;
  RTC_DCHECK(!unsent_on_chain_.none());
}

}  // namespace webrtc
