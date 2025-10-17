/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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
#include "api/candidate.h"

#include <string>

#include "p2p/base/p2p_constants.h"
#include "rtc_base/socket_address.h"
#include "test/gtest.h"

using webrtc::IceCandidateType;

namespace cricket {

TEST(CandidateTest, Id) {
  Candidate c;
  EXPECT_EQ(c.id().size(), 8u);
  std::string current_id = c.id();
  // Generate a new ID.
  c.generate_id();
  EXPECT_EQ(c.id().size(), 8u);
  EXPECT_NE(current_id, c.id());
}

TEST(CandidateTest, Component) {
  Candidate c;
  EXPECT_EQ(c.component(), ICE_CANDIDATE_COMPONENT_DEFAULT);
  c.set_component(ICE_CANDIDATE_COMPONENT_RTCP);
  EXPECT_EQ(c.component(), ICE_CANDIDATE_COMPONENT_RTCP);
}

TEST(CandidateTest, TypeName) {
  Candidate c;
  // The `type_name()` property defaults to "host".
  EXPECT_EQ(c.type_name(), "host");
  EXPECT_EQ(c.type(), IceCandidateType::kHost);

  c.set_type(IceCandidateType::kSrflx);
  EXPECT_EQ(c.type_name(), "srflx");
  EXPECT_EQ(c.type(), IceCandidateType::kSrflx);

  c.set_type(IceCandidateType::kPrflx);
  EXPECT_EQ(c.type_name(), "prflx");
  EXPECT_EQ(c.type(), IceCandidateType::kPrflx);

  c.set_type(IceCandidateType::kRelay);
  EXPECT_EQ(c.type_name(), "relay");
  EXPECT_EQ(c.type(), IceCandidateType::kRelay);
}

TEST(CandidateTest, Foundation) {
  Candidate c;
  EXPECT_TRUE(c.foundation().empty());
  c.set_protocol("udp");
  c.set_relay_protocol("udp");

  rtc::SocketAddress address("99.99.98.1", 1024);
  c.set_address(address);
  c.ComputeFoundation(c.address(), 1);
  std::string foundation1 = c.foundation();
  EXPECT_FALSE(foundation1.empty());

  // Change the tiebreaker.
  c.ComputeFoundation(c.address(), 2);
  std::string foundation2 = c.foundation();
  EXPECT_NE(foundation1, foundation2);

  // Provide a different base address.
  address.SetIP("100.100.100.1");
  c.ComputeFoundation(address, 1);  // Same tiebreaker as for foundation1.
  foundation2 = c.foundation();
  EXPECT_NE(foundation1, foundation2);

  // Consistency check (just in case the algorithm ever changes to random!).
  c.ComputeFoundation(c.address(), 1);
  foundation2 = c.foundation();
  EXPECT_EQ(foundation1, foundation2);

  // Changing the protocol should affect the foundation.
  auto prev_protocol = c.protocol();
  c.set_protocol("tcp");
  ASSERT_NE(prev_protocol, c.protocol());
  c.ComputeFoundation(c.address(), 1);
  EXPECT_NE(foundation1, c.foundation());
  c.set_protocol(prev_protocol);

  // Changing the relay protocol should affect the foundation.
  prev_protocol = c.relay_protocol();
  c.set_relay_protocol("tcp");
  ASSERT_NE(prev_protocol, c.relay_protocol());
  c.ComputeFoundation(c.address(), 1);
  EXPECT_NE(foundation1, c.foundation());
}

}  // namespace cricket
