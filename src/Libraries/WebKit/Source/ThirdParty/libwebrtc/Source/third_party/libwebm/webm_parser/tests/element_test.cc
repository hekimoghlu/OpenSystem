/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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

// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "webm/element.h"

#include <string>
#include <utility>

#include "gtest/gtest.h"

using webm::Element;

namespace {

class ElementTest : public testing::Test {};

TEST_F(ElementTest, Construction) {
  Element<int> value_initialized;
  EXPECT_EQ(false, value_initialized.is_present());
  EXPECT_EQ(0, value_initialized.value());

  Element<int> absent_custom_default(1);
  EXPECT_EQ(false, absent_custom_default.is_present());
  EXPECT_EQ(1, absent_custom_default.value());

  Element<int> present(2, true);
  EXPECT_EQ(true, present.is_present());
  EXPECT_EQ(2, present.value());
}

TEST_F(ElementTest, Assignment) {
  Element<int> e;

  e.Set(42, true);
  EXPECT_EQ(true, e.is_present());
  EXPECT_EQ(42, e.value());

  *e.mutable_value() = 0;
  EXPECT_EQ(true, e.is_present());
  EXPECT_EQ(0, e.value());
}

}  // namespace
