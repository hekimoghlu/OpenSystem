/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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
#include "net/dcsctp/packet/error_cause/user_initiated_abort_cause.h"

#include <stdint.h>

#include <type_traits>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/packet/error_cause/error_cause.h"
#include "net/dcsctp/packet/tlv_trait.h"
#include "net/dcsctp/testing/testing_macros.h"
#include "rtc_base/gunit.h"
#include "test/gmock.h"

namespace dcsctp {
namespace {
using ::testing::SizeIs;

TEST(UserInitiatedAbortCauseTest, EmptyReason) {
  Parameters causes =
      Parameters::Builder().Add(UserInitiatedAbortCause("")).Build();

  ASSERT_HAS_VALUE_AND_ASSIGN(Parameters deserialized,
                              Parameters::Parse(causes.data()));
  ASSERT_THAT(deserialized.descriptors(), SizeIs(1));
  EXPECT_EQ(deserialized.descriptors()[0].type, UserInitiatedAbortCause::kType);

  ASSERT_HAS_VALUE_AND_ASSIGN(
      UserInitiatedAbortCause cause,
      UserInitiatedAbortCause::Parse(deserialized.descriptors()[0].data));

  EXPECT_EQ(cause.upper_layer_abort_reason(), "");
}

TEST(UserInitiatedAbortCauseTest, SetReason) {
  Parameters causes = Parameters::Builder()
                          .Add(UserInitiatedAbortCause("User called Close"))
                          .Build();

  ASSERT_HAS_VALUE_AND_ASSIGN(Parameters deserialized,
                              Parameters::Parse(causes.data()));
  ASSERT_THAT(deserialized.descriptors(), SizeIs(1));
  EXPECT_EQ(deserialized.descriptors()[0].type, UserInitiatedAbortCause::kType);

  ASSERT_HAS_VALUE_AND_ASSIGN(
      UserInitiatedAbortCause cause,
      UserInitiatedAbortCause::Parse(deserialized.descriptors()[0].data));

  EXPECT_EQ(cause.upper_layer_abort_reason(), "User called Close");
}

}  // namespace
}  // namespace dcsctp
