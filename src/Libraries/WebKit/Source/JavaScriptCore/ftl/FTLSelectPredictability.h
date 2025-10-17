/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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
#pragma once

#if ENABLE(FTL_JIT)

namespace JSC::FTL {

enum class SelectPredictability : uint8_t {
    // Use this when we expect it to be very unlikely the branch predictor will be able to guess which side of the select will be chosen. This tells B3 to try to emit the select as a conditional move, which (usually) is not speculated by the CPU.
    NotPredictable,
    // Use this when it's possible a branch predictor will do a good job guessing the selected value but we don't know a priori which side is more likely.
    Predictable,
    // Use these when we are very sure one of the two sides is substantially more likely than the other.
    LeftLikely,
    RightLikely,
};

} // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
