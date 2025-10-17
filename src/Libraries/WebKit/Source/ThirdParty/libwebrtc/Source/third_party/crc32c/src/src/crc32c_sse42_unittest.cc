/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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

// Copyright 2017 The CRC32C Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "gtest/gtest.h"

#include "./crc32c_extend_unittests.h"
#include "./crc32c_sse42.h"

namespace crc32c {

#if HAVE_SSE42 && (defined(_M_X64) || defined(__x86_64__))

struct Sse42TestTraits {
  static uint32_t Extend(uint32_t crc, const uint8_t* data, size_t count) {
    return ExtendSse42(crc, data, count);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Sse42, ExtendTest, Sse42TestTraits);

#endif  // HAVE_SSE42 && (defined(_M_X64) || defined(__x86_64__))

}  // namespace crc32c
