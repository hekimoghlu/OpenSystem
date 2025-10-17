/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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

#include <wtf/OptionSet.h>

namespace WebCore {
namespace Style {

enum class CascadeLevel : uint8_t {
    UserAgent   = 1 << 0,
    User        = 1 << 1,
    Author      = 1 << 2
};

inline CascadeLevel& operator--(CascadeLevel& level)
{
    switch (level) {
    case CascadeLevel::Author:
        return level = CascadeLevel::User;
    case CascadeLevel::User:
        return level = CascadeLevel::UserAgent;
    case CascadeLevel::UserAgent:
        ASSERT_NOT_REACHED();
        return level;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
}
