/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
#ifndef RTC_BASE_FAKE_MDNS_RESPONDER_H_
#define RTC_BASE_FAKE_MDNS_RESPONDER_H_

#include <map>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "rtc_base/ip_address.h"
#include "rtc_base/mdns_responder_interface.h"
#include "rtc_base/thread.h"

namespace webrtc {

// This class posts tasks on the given `thread` to invoke callbacks. It's the
// callback's responsibility to be aware of potential destruction of state it
// depends on, e.g., using WeakPtrFactory or PendingTaskSafetyFlag.
class FakeMdnsResponder : public MdnsResponderInterface {
 public:
  explicit FakeMdnsResponder(rtc::Thread* thread) : thread_(thread) {}
  ~FakeMdnsResponder() = default;

  void CreateNameForAddress(const rtc::IPAddress& addr,
                            NameCreatedCallback callback) override {
    std::string name;
    if (addr_name_map_.find(addr) != addr_name_map_.end()) {
      name = addr_name_map_[addr];
    } else {
      name = std::to_string(next_available_id_++) + ".local";
      addr_name_map_[addr] = name;
    }
    thread_->PostTask([callback, addr, name]() { callback(addr, name); });
  }
  void RemoveNameForAddress(const rtc::IPAddress& addr,
                            NameRemovedCallback callback) override {
    auto it = addr_name_map_.find(addr);
    if (it != addr_name_map_.end()) {
      addr_name_map_.erase(it);
    }
    bool result = it != addr_name_map_.end();
    thread_->PostTask([callback, result]() { callback(result); });
  }

  rtc::IPAddress GetMappedAddressForName(absl::string_view name) const {
    for (const auto& addr_name_pair : addr_name_map_) {
      if (addr_name_pair.second == name) {
        return addr_name_pair.first;
      }
    }
    return rtc::IPAddress();
  }

 private:
  uint32_t next_available_id_ = 0;
  std::map<rtc::IPAddress, std::string> addr_name_map_;
  rtc::Thread* const thread_;
};

}  // namespace webrtc

#endif  // RTC_BASE_FAKE_MDNS_RESPONDER_H_
