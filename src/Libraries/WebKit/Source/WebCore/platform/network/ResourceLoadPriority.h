/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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

#include <wtf/EnumTraits.h>

namespace WebCore {

enum class ResourceLoadPriority : uint8_t {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
    Lowest = VeryLow,
    Highest = VeryHigh,
};
static constexpr unsigned bitWidthOfResourceLoadPriority = 3;
static_assert(static_cast<unsigned>(ResourceLoadPriority::Highest) <= ((1U << bitWidthOfResourceLoadPriority) - 1));

static const unsigned resourceLoadPriorityCount { static_cast<unsigned>(ResourceLoadPriority::Highest) + 1 };

inline ResourceLoadPriority& operator++(ResourceLoadPriority& priority)
{
    ASSERT(priority != ResourceLoadPriority::Highest);
    return priority = static_cast<ResourceLoadPriority>(static_cast<int>(priority) + 1);
}

inline ResourceLoadPriority& operator--(ResourceLoadPriority& priority)
{
    ASSERT(priority != ResourceLoadPriority::Lowest);
    return priority = static_cast<ResourceLoadPriority>(static_cast<int>(priority) - 1);
}

} // namespace WebCore

namespace WTF {

template<> struct EnumTraitsForPersistence<WebCore::ResourceLoadPriority> {
    using values = EnumValues<
        WebCore::ResourceLoadPriority,
        WebCore::ResourceLoadPriority::VeryLow,
        WebCore::ResourceLoadPriority::Low,
        WebCore::ResourceLoadPriority::Medium,
        WebCore::ResourceLoadPriority::High,
        WebCore::ResourceLoadPriority::VeryHigh
    >;
};

} // namespace WTF
