/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#include "pc/rtcp_mux_filter.h"

#include "test/gtest.h"

TEST(RtcpMuxFilterTest, IsActiveSender) {
  cricket::RtcpMuxFilter filter;
  // Init state - not active
  EXPECT_FALSE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_FALSE(filter.IsFullyActive());
  // After sent offer, demux should not be active.
  filter.SetOffer(true, cricket::CS_LOCAL);
  EXPECT_FALSE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_FALSE(filter.IsFullyActive());
  // Remote accepted, filter is now active.
  filter.SetAnswer(true, cricket::CS_REMOTE);
  EXPECT_TRUE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_TRUE(filter.IsFullyActive());
}

// Test that we can receive provisional answer and final answer.
TEST(RtcpMuxFilterTest, ReceivePrAnswer) {
  cricket::RtcpMuxFilter filter;
  filter.SetOffer(true, cricket::CS_LOCAL);
  // Received provisional answer with mux enabled.
  EXPECT_TRUE(filter.SetProvisionalAnswer(true, cricket::CS_REMOTE));
  // We are now provisionally active since both sender and receiver support mux.
  EXPECT_TRUE(filter.IsActive());
  EXPECT_TRUE(filter.IsProvisionallyActive());
  EXPECT_FALSE(filter.IsFullyActive());
  // Received provisional answer with mux disabled.
  EXPECT_TRUE(filter.SetProvisionalAnswer(false, cricket::CS_REMOTE));
  // We are now inactive since the receiver doesn't support mux.
  EXPECT_FALSE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_FALSE(filter.IsFullyActive());
  // Received final answer with mux enabled.
  EXPECT_TRUE(filter.SetAnswer(true, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_TRUE(filter.IsFullyActive());
}

TEST(RtcpMuxFilterTest, IsActiveReceiver) {
  cricket::RtcpMuxFilter filter;
  // Init state - not active.
  EXPECT_FALSE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_FALSE(filter.IsFullyActive());
  // After received offer, demux should not be active
  filter.SetOffer(true, cricket::CS_REMOTE);
  EXPECT_FALSE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_FALSE(filter.IsFullyActive());
  // We accept, filter is now active
  filter.SetAnswer(true, cricket::CS_LOCAL);
  EXPECT_TRUE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_TRUE(filter.IsFullyActive());
}

// Test that we can send provisional answer and final answer.
TEST(RtcpMuxFilterTest, SendPrAnswer) {
  cricket::RtcpMuxFilter filter;
  filter.SetOffer(true, cricket::CS_REMOTE);
  // Send provisional answer with mux enabled.
  EXPECT_TRUE(filter.SetProvisionalAnswer(true, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());
  EXPECT_TRUE(filter.IsProvisionallyActive());
  EXPECT_FALSE(filter.IsFullyActive());
  // Received provisional answer with mux disabled.
  EXPECT_TRUE(filter.SetProvisionalAnswer(false, cricket::CS_LOCAL));
  EXPECT_FALSE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_FALSE(filter.IsFullyActive());
  // Send final answer with mux enabled.
  EXPECT_TRUE(filter.SetAnswer(true, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());
  EXPECT_FALSE(filter.IsProvisionallyActive());
  EXPECT_TRUE(filter.IsFullyActive());
}

// Test that we can enable the filter in an update.
// We can not disable the filter later since that would mean we need to
// recreate a rtcp transport channel.
TEST(RtcpMuxFilterTest, EnableFilterDuringUpdate) {
  cricket::RtcpMuxFilter filter;
  EXPECT_FALSE(filter.IsActive());
  EXPECT_TRUE(filter.SetOffer(false, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.SetAnswer(false, cricket::CS_LOCAL));
  EXPECT_FALSE(filter.IsActive());

  EXPECT_TRUE(filter.SetOffer(true, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.SetAnswer(true, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());

  EXPECT_FALSE(filter.SetOffer(false, cricket::CS_REMOTE));
  EXPECT_FALSE(filter.SetAnswer(false, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());
}

// Test that SetOffer can be called twice.
TEST(RtcpMuxFilterTest, SetOfferTwice) {
  cricket::RtcpMuxFilter filter;

  EXPECT_TRUE(filter.SetOffer(true, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.SetOffer(true, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.SetAnswer(true, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());

  cricket::RtcpMuxFilter filter2;
  EXPECT_TRUE(filter2.SetOffer(false, cricket::CS_LOCAL));
  EXPECT_TRUE(filter2.SetOffer(false, cricket::CS_LOCAL));
  EXPECT_TRUE(filter2.SetAnswer(false, cricket::CS_REMOTE));
  EXPECT_FALSE(filter2.IsActive());
}

// Test that the filter can be enabled twice.
TEST(RtcpMuxFilterTest, EnableFilterTwiceDuringUpdate) {
  cricket::RtcpMuxFilter filter;

  EXPECT_TRUE(filter.SetOffer(true, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.SetAnswer(true, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());

  EXPECT_TRUE(filter.SetOffer(true, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.SetAnswer(true, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());
}

// Test that the filter can be kept disabled during updates.
TEST(RtcpMuxFilterTest, KeepFilterDisabledDuringUpdate) {
  cricket::RtcpMuxFilter filter;

  EXPECT_TRUE(filter.SetOffer(false, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.SetAnswer(false, cricket::CS_LOCAL));
  EXPECT_FALSE(filter.IsActive());

  EXPECT_TRUE(filter.SetOffer(false, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.SetAnswer(false, cricket::CS_LOCAL));
  EXPECT_FALSE(filter.IsActive());
}

// Test that we can SetActive and then can't deactivate.
TEST(RtcpMuxFilterTest, SetActiveCantDeactivate) {
  cricket::RtcpMuxFilter filter;

  filter.SetActive();
  EXPECT_TRUE(filter.IsActive());

  EXPECT_FALSE(filter.SetOffer(false, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());
  EXPECT_TRUE(filter.SetOffer(true, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());

  EXPECT_FALSE(filter.SetProvisionalAnswer(false, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.IsActive());
  EXPECT_TRUE(filter.SetProvisionalAnswer(true, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.IsActive());

  EXPECT_FALSE(filter.SetAnswer(false, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.IsActive());
  EXPECT_TRUE(filter.SetAnswer(true, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.IsActive());

  EXPECT_FALSE(filter.SetOffer(false, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.IsActive());
  EXPECT_TRUE(filter.SetOffer(true, cricket::CS_REMOTE));
  EXPECT_TRUE(filter.IsActive());

  EXPECT_FALSE(filter.SetProvisionalAnswer(false, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());
  EXPECT_TRUE(filter.SetProvisionalAnswer(true, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());

  EXPECT_FALSE(filter.SetAnswer(false, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());
  EXPECT_TRUE(filter.SetAnswer(true, cricket::CS_LOCAL));
  EXPECT_TRUE(filter.IsActive());
}
