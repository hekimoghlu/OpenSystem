/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
#include "config.h"
#include <wtf/unicode/icu/ICUHelpers.h>

#include <mutex>
#include <span>
#include <unicode/uvernum.h>

namespace WTF {
namespace ICU {

static std::span<const uint8_t, U_MAX_VERSION_LENGTH> version()
{
    static UVersionInfo versions { };
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        u_getVersion(versions);
    });
    return std::span { versions };
}

unsigned majorVersion()
{
    static_assert(0 < U_MAX_VERSION_LENGTH);
    return version()[0];
}

unsigned minorVersion()
{
    static_assert(1 < U_MAX_VERSION_LENGTH);
    return version()[1];
}

} } // namespace WTF::ICU
