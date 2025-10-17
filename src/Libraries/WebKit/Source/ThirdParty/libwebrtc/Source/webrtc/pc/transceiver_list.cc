/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 17, 2023.
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
#include "pc/transceiver_list.h"

#include <string>

#include "rtc_base/checks.h"

namespace webrtc {

void TransceiverStableState::set_newly_created() {
  RTC_DCHECK(!has_m_section_);
  newly_created_ = true;
}

void TransceiverStableState::SetMSectionIfUnset(
    std::optional<std::string> mid,
    std::optional<size_t> mline_index) {
  if (!has_m_section_) {
    mid_ = mid;
    mline_index_ = mline_index;
    has_m_section_ = true;
  }
}

void TransceiverStableState::SetRemoteStreamIds(
    const std::vector<std::string>& ids) {
  if (!remote_stream_ids_.has_value()) {
    remote_stream_ids_ = ids;
  }
}

void TransceiverStableState::SetInitSendEncodings(
    const std::vector<RtpEncodingParameters>& encodings) {
  init_send_encodings_ = encodings;
}

std::vector<RtpTransceiver*> TransceiverList::ListInternal() const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  std::vector<RtpTransceiver*> internals;
  for (auto transceiver : transceivers_) {
    internals.push_back(transceiver->internal());
  }
  return internals;
}

RtpTransceiverProxyRefPtr TransceiverList::FindBySender(
    rtc::scoped_refptr<RtpSenderInterface> sender) const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  for (auto transceiver : transceivers_) {
    if (transceiver->sender() == sender) {
      return transceiver;
    }
  }
  return nullptr;
}

RtpTransceiverProxyRefPtr TransceiverList::FindByMid(
    const std::string& mid) const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  for (auto transceiver : transceivers_) {
    if (transceiver->mid() == mid) {
      return transceiver;
    }
  }
  return nullptr;
}

RtpTransceiverProxyRefPtr TransceiverList::FindByMLineIndex(
    size_t mline_index) const {
  RTC_DCHECK_RUN_ON(&sequence_checker_);
  for (auto transceiver : transceivers_) {
    if (transceiver->internal()->mline_index() == mline_index) {
      return transceiver;
    }
  }
  return nullptr;
}

}  // namespace webrtc
