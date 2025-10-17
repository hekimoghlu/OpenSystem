/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#include "api/adaptation/resource.h"

#include <memory>

#include "api/scoped_refptr.h"
#include "call/adaptation/test/fake_resource.h"
#include "call/adaptation/test/mock_resource_listener.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

using ::testing::_;
using ::testing::StrictMock;

class ResourceTest : public ::testing::Test {
 public:
  ResourceTest() : fake_resource_(FakeResource::Create("FakeResource")) {}

 protected:
  rtc::scoped_refptr<FakeResource> fake_resource_;
};

TEST_F(ResourceTest, RegisteringListenerReceivesCallbacks) {
  StrictMock<MockResourceListener> resource_listener;
  fake_resource_->SetResourceListener(&resource_listener);
  EXPECT_CALL(resource_listener, OnResourceUsageStateMeasured(_, _))
      .Times(1)
      .WillOnce([](rtc::scoped_refptr<Resource> /* resource */,
                   ResourceUsageState usage_state) {
        EXPECT_EQ(ResourceUsageState::kOveruse, usage_state);
      });
  fake_resource_->SetUsageState(ResourceUsageState::kOveruse);
  fake_resource_->SetResourceListener(nullptr);
}

TEST_F(ResourceTest, UnregisteringListenerStopsCallbacks) {
  StrictMock<MockResourceListener> resource_listener;
  fake_resource_->SetResourceListener(&resource_listener);
  fake_resource_->SetResourceListener(nullptr);
  EXPECT_CALL(resource_listener, OnResourceUsageStateMeasured(_, _)).Times(0);
  fake_resource_->SetUsageState(ResourceUsageState::kOveruse);
}

}  // namespace webrtc
