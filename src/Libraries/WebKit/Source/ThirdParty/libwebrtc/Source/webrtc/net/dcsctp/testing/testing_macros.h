/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
#ifndef NET_DCSCTP_TESTING_TESTING_MACROS_H_
#define NET_DCSCTP_TESTING_TESTING_MACROS_H_

#include <utility>

namespace dcsctp {

#define DCSCTP_CONCAT_INNER_(x, y) x##y
#define DCSCTP_CONCAT_(x, y) DCSCTP_CONCAT_INNER_(x, y)

// Similar to ASSERT_OK_AND_ASSIGN, this works with an std::optional<> instead
// of an absl::StatusOr<>.
#define ASSERT_HAS_VALUE_AND_ASSIGN(lhs, rexpr)                     \
  auto DCSCTP_CONCAT_(tmp_opt_val__, __LINE__) = rexpr;             \
  ASSERT_TRUE(DCSCTP_CONCAT_(tmp_opt_val__, __LINE__).has_value()); \
  lhs = *std::move(DCSCTP_CONCAT_(tmp_opt_val__, __LINE__));

}  // namespace dcsctp

#endif  // NET_DCSCTP_TESTING_TESTING_MACROS_H_
