/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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
#ifndef P2P_BASE_TEST_TURN_CUSTOMIZER_H_
#define P2P_BASE_TEST_TURN_CUSTOMIZER_H_

#include <memory>

#include "api/turn_customizer.h"
#include "rtc_base/gunit.h"

namespace cricket {

class TestTurnCustomizer : public webrtc::TurnCustomizer {
 public:
  TestTurnCustomizer() {}
  virtual ~TestTurnCustomizer() {}

  enum TestTurnAttributeExtensions {
    // Test only attribute
    STUN_ATTR_COUNTER = 0xFF02  // Number
  };

  void MaybeModifyOutgoingStunMessage(cricket::PortInterface* port,
                                      cricket::StunMessage* message) override {
    modify_cnt_++;

    ASSERT_NE(0, message->type());
    if (add_counter_) {
      message->AddAttribute(std::make_unique<cricket::StunUInt32Attribute>(
          STUN_ATTR_COUNTER, modify_cnt_));
    }
    return;
  }

  bool AllowChannelData(cricket::PortInterface* port,
                        const void* data,
                        size_t size,
                        bool payload) override {
    allow_channel_data_cnt_++;
    return allow_channel_data_;
  }

  bool add_counter_ = false;
  bool allow_channel_data_ = true;
  unsigned int modify_cnt_ = 0;
  unsigned int allow_channel_data_cnt_ = 0;
};

}  // namespace cricket

#endif  // P2P_BASE_TEST_TURN_CUSTOMIZER_H_
