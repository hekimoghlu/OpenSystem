/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#include "rtc_base/callback_list.h"

#include "rtc_base/checks.h"

namespace webrtc {
namespace callback_list_impl {

CallbackListReceivers::CallbackListReceivers() = default;

CallbackListReceivers::~CallbackListReceivers() {
  RTC_CHECK(!send_in_progress_);
}

void CallbackListReceivers::RemoveReceivers(const void* removal_tag) {
  RTC_DCHECK(removal_tag);

  // We divide the receivers_ vector into three regions: from right to left, the
  // "keep" region, the "todo" region, and the "remove" region. The "todo"
  // region initially covers the whole vector.
  size_t first_todo = 0;                    // First element of the "todo"
                                            // region.
  size_t first_remove = receivers_.size();  // First element of the "remove"
                                            // region.

  // Loop until the "todo" region is empty.
  while (first_todo != first_remove) {
    if (receivers_[first_todo].removal_tag != removal_tag) {
      // The first element of the "todo" region should be kept. Move the
      // "keep"/"todo" boundary.
      ++first_todo;
    } else if (receivers_[first_remove - 1].removal_tag == removal_tag) {
      // The last element of the "todo" region should be removed. Move the
      // "todo"/"remove" boundary.
      if (send_in_progress_) {
        // Tag this receiver for removal, which will be done when `ForEach`
        // has completed.
        receivers_[first_remove - 1].removal_tag = pending_removal_tag();
      }
      --first_remove;
    } else if (!send_in_progress_) {
      // The first element of the "todo" region should be removed, and the last
      // element of the "todo" region should be kept. Swap them, and then shrink
      // the "todo" region from both ends.
      RTC_DCHECK_NE(first_todo, first_remove - 1);
      using std::swap;
      swap(receivers_[first_todo], receivers_[first_remove - 1]);
      RTC_DCHECK_NE(receivers_[first_todo].removal_tag, removal_tag);
      ++first_todo;
      RTC_DCHECK_EQ(receivers_[first_remove - 1].removal_tag, removal_tag);
      --first_remove;
    }
  }

  if (!send_in_progress_) {
    // Discard the remove region.
    receivers_.resize(first_remove);
  }
}

void CallbackListReceivers::Foreach(
    rtc::FunctionView<void(UntypedFunction&)> fv) {
  RTC_CHECK(!send_in_progress_);
  bool removals_detected = false;
  send_in_progress_ = true;
  for (auto& r : receivers_) {
    RTC_DCHECK_NE(r.removal_tag, pending_removal_tag());
    fv(r.function);
    if (r.removal_tag == pending_removal_tag()) {
      removals_detected = true;
    }
  }
  send_in_progress_ = false;
  if (removals_detected) {
    RemoveReceivers(pending_removal_tag());
  }
}

template void CallbackListReceivers::AddReceiver(
    const void*,
    UntypedFunction::TrivialUntypedFunctionArgs<1>);
template void CallbackListReceivers::AddReceiver(
    const void*,
    UntypedFunction::TrivialUntypedFunctionArgs<2>);
template void CallbackListReceivers::AddReceiver(
    const void*,
    UntypedFunction::TrivialUntypedFunctionArgs<3>);
template void CallbackListReceivers::AddReceiver(
    const void*,
    UntypedFunction::TrivialUntypedFunctionArgs<4>);
template void CallbackListReceivers::AddReceiver(
    const void*,
    UntypedFunction::NontrivialUntypedFunctionArgs);
template void CallbackListReceivers::AddReceiver(
    const void*,
    UntypedFunction::FunctionPointerUntypedFunctionArgs);

template void CallbackListReceivers::AddReceiver(
    UntypedFunction::TrivialUntypedFunctionArgs<1>);
template void CallbackListReceivers::AddReceiver(
    UntypedFunction::TrivialUntypedFunctionArgs<2>);
template void CallbackListReceivers::AddReceiver(
    UntypedFunction::TrivialUntypedFunctionArgs<3>);
template void CallbackListReceivers::AddReceiver(
    UntypedFunction::TrivialUntypedFunctionArgs<4>);
template void CallbackListReceivers::AddReceiver(
    UntypedFunction::NontrivialUntypedFunctionArgs);
template void CallbackListReceivers::AddReceiver(
    UntypedFunction::FunctionPointerUntypedFunctionArgs);

}  // namespace callback_list_impl
}  // namespace webrtc
