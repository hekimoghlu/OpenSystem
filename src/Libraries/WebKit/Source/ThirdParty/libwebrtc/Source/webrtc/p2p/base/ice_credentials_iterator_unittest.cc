/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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
#include "p2p/base/ice_credentials_iterator.h"

#include <vector>

#include "test/gtest.h"

using cricket::IceCredentialsIterator;
using cricket::IceParameters;

TEST(IceCredentialsIteratorTest, GetEmpty) {
  std::vector<IceParameters> empty;
  IceCredentialsIterator iterator(empty);
  // Verify that we can get credentials even if input is empty.
  IceParameters credentials1 = iterator.GetIceCredentials();
}

TEST(IceCredentialsIteratorTest, GetOne) {
  std::vector<IceParameters> one = {
      IceCredentialsIterator::CreateRandomIceCredentials()};
  IceCredentialsIterator iterator(one);
  EXPECT_EQ(iterator.GetIceCredentials(), one[0]);
  auto random = iterator.GetIceCredentials();
  EXPECT_NE(random, one[0]);
  EXPECT_NE(random, iterator.GetIceCredentials());
}

TEST(IceCredentialsIteratorTest, GetTwo) {
  std::vector<IceParameters> two = {
      IceCredentialsIterator::CreateRandomIceCredentials(),
      IceCredentialsIterator::CreateRandomIceCredentials()};
  IceCredentialsIterator iterator(two);
  EXPECT_EQ(iterator.GetIceCredentials(), two[1]);
  EXPECT_EQ(iterator.GetIceCredentials(), two[0]);
  auto random = iterator.GetIceCredentials();
  EXPECT_NE(random, two[0]);
  EXPECT_NE(random, two[1]);
  EXPECT_NE(random, iterator.GetIceCredentials());
}
