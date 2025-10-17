/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#include "call/adaptation/broadcast_resource_listener.h"

#include "call/adaptation/test/fake_resource.h"
#include "call/adaptation/test/mock_resource_listener.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace webrtc {

using ::testing::_;
using ::testing::StrictMock;

TEST(BroadcastResourceListenerTest, CreateAndRemoveAdapterResource) {
  rtc::scoped_refptr<FakeResource> source_resource =
      FakeResource::Create("SourceResource");
  BroadcastResourceListener broadcast_resource_listener(source_resource);
  broadcast_resource_listener.StartListening();

  EXPECT_TRUE(broadcast_resource_listener.GetAdapterResources().empty());
  rtc::scoped_refptr<Resource> adapter =
      broadcast_resource_listener.CreateAdapterResource();
  StrictMock<MockResourceListener> listener;
  adapter->SetResourceListener(&listener);
  EXPECT_EQ(std::vector<rtc::scoped_refptr<Resource>>{adapter},
            broadcast_resource_listener.GetAdapterResources());

  // The removed adapter is not referenced by the broadcaster.
  broadcast_resource_listener.RemoveAdapterResource(adapter);
  EXPECT_TRUE(broadcast_resource_listener.GetAdapterResources().empty());
  // The removed adapter is not forwarding measurements.
  EXPECT_CALL(listener, OnResourceUsageStateMeasured(_, _)).Times(0);
  source_resource->SetUsageState(ResourceUsageState::kOveruse);
  // Cleanup.
  adapter->SetResourceListener(nullptr);
  broadcast_resource_listener.StopListening();
}

TEST(BroadcastResourceListenerTest, AdapterNameIsBasedOnSourceResourceName) {
  rtc::scoped_refptr<FakeResource> source_resource =
      FakeResource::Create("FooBarResource");
  BroadcastResourceListener broadcast_resource_listener(source_resource);
  broadcast_resource_listener.StartListening();

  rtc::scoped_refptr<Resource> adapter =
      broadcast_resource_listener.CreateAdapterResource();
  EXPECT_EQ("FooBarResourceAdapter", adapter->Name());

  broadcast_resource_listener.RemoveAdapterResource(adapter);
  broadcast_resource_listener.StopListening();
}

TEST(BroadcastResourceListenerTest, AdaptersForwardsUsageMeasurements) {
  rtc::scoped_refptr<FakeResource> source_resource =
      FakeResource::Create("SourceResource");
  BroadcastResourceListener broadcast_resource_listener(source_resource);
  broadcast_resource_listener.StartListening();

  StrictMock<MockResourceListener> destination_listener1;
  StrictMock<MockResourceListener> destination_listener2;
  rtc::scoped_refptr<Resource> adapter1 =
      broadcast_resource_listener.CreateAdapterResource();
  adapter1->SetResourceListener(&destination_listener1);
  rtc::scoped_refptr<Resource> adapter2 =
      broadcast_resource_listener.CreateAdapterResource();
  adapter2->SetResourceListener(&destination_listener2);

  // Expect kOveruse to be echoed.
  EXPECT_CALL(destination_listener1, OnResourceUsageStateMeasured(_, _))
      .Times(1)
      .WillOnce([adapter1](rtc::scoped_refptr<Resource> resource,
                           ResourceUsageState usage_state) {
        EXPECT_EQ(adapter1, resource);
        EXPECT_EQ(ResourceUsageState::kOveruse, usage_state);
      });
  EXPECT_CALL(destination_listener2, OnResourceUsageStateMeasured(_, _))
      .Times(1)
      .WillOnce([adapter2](rtc::scoped_refptr<Resource> resource,
                           ResourceUsageState usage_state) {
        EXPECT_EQ(adapter2, resource);
        EXPECT_EQ(ResourceUsageState::kOveruse, usage_state);
      });
  source_resource->SetUsageState(ResourceUsageState::kOveruse);

  // Expect kUnderuse to be echoed.
  EXPECT_CALL(destination_listener1, OnResourceUsageStateMeasured(_, _))
      .Times(1)
      .WillOnce([adapter1](rtc::scoped_refptr<Resource> resource,
                           ResourceUsageState usage_state) {
        EXPECT_EQ(adapter1, resource);
        EXPECT_EQ(ResourceUsageState::kUnderuse, usage_state);
      });
  EXPECT_CALL(destination_listener2, OnResourceUsageStateMeasured(_, _))
      .Times(1)
      .WillOnce([adapter2](rtc::scoped_refptr<Resource> resource,
                           ResourceUsageState usage_state) {
        EXPECT_EQ(adapter2, resource);
        EXPECT_EQ(ResourceUsageState::kUnderuse, usage_state);
      });
  source_resource->SetUsageState(ResourceUsageState::kUnderuse);

  // Adapters have to be unregistered before they or the broadcaster is
  // destroyed, ensuring safe use of raw pointers.
  adapter1->SetResourceListener(nullptr);
  adapter2->SetResourceListener(nullptr);

  broadcast_resource_listener.RemoveAdapterResource(adapter1);
  broadcast_resource_listener.RemoveAdapterResource(adapter2);
  broadcast_resource_listener.StopListening();
}

}  // namespace webrtc
