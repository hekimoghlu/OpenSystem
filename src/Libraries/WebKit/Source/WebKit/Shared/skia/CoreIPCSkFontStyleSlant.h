/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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

#if USE(SKIA)

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkFontStyle.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebKit {

class CoreIPCSkFontStyleSlant {
public:
    CoreIPCSkFontStyleSlant(SkFontStyle::Slant slant)
        : m_slant(slant)
    {
    }

    static SkFontStyle::Slant create(uint8_t slant)
    {
        return static_cast<SkFontStyle::Slant>(slant);
    }

    uint8_t data() const
    {
        return m_slant;
    }

private:
    const SkFontStyle::Slant m_slant;
};

} // namespace WebKit

#endif // USE(SKIA)
