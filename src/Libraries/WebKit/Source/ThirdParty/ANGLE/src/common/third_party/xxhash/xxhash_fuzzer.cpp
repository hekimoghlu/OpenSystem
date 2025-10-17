/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 16, 2022.
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

//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// xxHash Fuzzer test:
//      Integration with Chromium's libfuzzer for xxHash.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "xxhash.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
#if !defined(XXH_NO_LONG_LONG)
    // Test 64-bit hash.
    unsigned long long seed64 = 0ull;
    size_t seedSize64         = sizeof(seed64);
    if (size >= seedSize64)
    {
        memcpy(&seed64, data, seedSize64);
    }
    else
    {
        seedSize64 = 0;
    }
    XXH64(&data[seedSize64], size - seedSize64, seed64);
#endif  // !defined(XXH_NO_LONG_LONG)

    // Test 32-bit hash.
    unsigned int seed32 = 0u;
    size_t seedSize32   = sizeof(seed32);
    if (size >= seedSize32)
    {
        memcpy(&seed32, data, seedSize32);
    }
    else
    {
        seedSize32 = 0;
    }
    XXH32(&data[seedSize32], size - seedSize32, seed32);
    return 0;
}
