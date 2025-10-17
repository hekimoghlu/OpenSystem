/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#ifndef API_TEST_MOCK_DTMF_SENDER_H_
#define API_TEST_MOCK_DTMF_SENDER_H_

#include <string>
#include <type_traits>

#include "api/dtmf_sender_interface.h"
#include "api/make_ref_counted.h"
#include "api/scoped_refptr.h"
#include "rtc_base/ref_counted_object.h"
#include "test/gmock.h"

namespace webrtc {

class MockDtmfSenderObserver : public DtmfSenderObserverInterface {
 public:
  MOCK_METHOD(void,
              OnToneChange,
              (const std::string&, const std::string&),
              (override));
  MOCK_METHOD(void, OnToneChange, (const std::string&), (override));
};

static_assert(!std::is_abstract_v<MockDtmfSenderObserver>, "");

class MockDtmfSender : public DtmfSenderInterface {
 public:
  static rtc::scoped_refptr<MockDtmfSender> Create() {
    return rtc::make_ref_counted<MockDtmfSender>();
  }

  MOCK_METHOD(void,
              RegisterObserver,
              (DtmfSenderObserverInterface * observer),
              (override));
  MOCK_METHOD(void, UnregisterObserver, (), (override));
  MOCK_METHOD(bool, CanInsertDtmf, (), (override));
  MOCK_METHOD(std::string, tones, (), (const, override));
  MOCK_METHOD(int, duration, (), (const, override));
  MOCK_METHOD(int, inter_tone_gap, (), (const, override));

 protected:
  MockDtmfSender() = default;
};

static_assert(!std::is_abstract_v<rtc::RefCountedObject<MockDtmfSender>>, "");

}  // namespace webrtc

#endif  // API_TEST_MOCK_DTMF_SENDER_H_
