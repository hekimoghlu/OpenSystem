/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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

#include <stdint.h>

namespace WTF {

template<typename IntegerType> IntegerType cryptographicallyRandomNumber() = delete;

template<> WTF_EXPORT_PRIVATE uint8_t cryptographicallyRandomNumber<uint8_t>();

// Returns a cryptographically secure pseudo-random number in the range [0, UINT_MAX].
template<> WTF_EXPORT_PRIVATE unsigned cryptographicallyRandomNumber<unsigned>();

// Returns a cryptographically secure pseudo-random number in the range [0, UINT64_MAX].
template<> WTF_EXPORT_PRIVATE uint64_t cryptographicallyRandomNumber<uint64_t>();

WTF_EXPORT_PRIVATE void cryptographicallyRandomValues(std::span<uint8_t>);

// Returns a cryptographically secure pseudo-random number in the range [0, 1), with 32 bits of randomness.
WTF_EXPORT_PRIVATE double cryptographicallyRandomUnitInterval();

}

using WTF::cryptographicallyRandomNumber;
using WTF::cryptographicallyRandomUnitInterval;
using WTF::cryptographicallyRandomValues;
