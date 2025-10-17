/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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
#include <skia/core/SkString.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#include <wtf/StdLibExtras.h>

namespace WebKit {

class CoreIPCSkString {
public:
    CoreIPCSkString(const SkString& string)
        : m_string(string)
    {
    }

    static SkString create(std::span<const char> span)
    {
        return { span.data(), span.size() };
    }

    std::span<const char> data() const
    {
        return unsafeMakeSpan(m_string.data(), m_string.size());
    }

private:
    const SkString& m_string;
};

} // namespace WebKit

#endif // USE(SKIA)
