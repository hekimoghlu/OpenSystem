/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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

#if USE(CG)
#include <wtf/RetainPtr.h>
typedef struct CGColorSpace* CGColorSpaceRef;
#elif USE(SKIA)
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkColorSpace.h>
#include <skia/core/SkData.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#else
#include <optional>
#endif

namespace WebCore {

#if USE(CG)

using PlatformColorSpace = RetainPtr<CGColorSpaceRef>;
using PlatformColorSpaceValue = CGColorSpaceRef;

#elif USE(SKIA)

using PlatformColorSpace = sk_sp<SkColorSpace>;
using PlatformColorSpaceValue = sk_sp<SkColorSpace>;

#else

class PlatformColorSpace {
public:
    enum class Name : uint8_t {
        SRGB
        , LinearSRGB
#if ENABLE(DESTINATION_COLOR_SPACE_DISPLAY_P3)
        , DisplayP3
#endif
    };

    PlatformColorSpace(Name name)
        : m_name { name }
    {
    }

    Name get() const { return m_name; }

private:
    Name m_name;

};
using PlatformColorSpaceValue = PlatformColorSpace::Name;

#endif

}
