/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // Skia port
#include <skia/core/SkColorSpace.h>
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

namespace WebKit {

class CoreIPCSkColorSpace {
public:
    explicit CoreIPCSkColorSpace(sk_sp<SkColorSpace> skColorSpace)
        : m_skColorSpace(WTFMove(skColorSpace))
    {
    }

    static sk_sp<SkColorSpace> create(std::span<const uint8_t> data)
    {
        return SkColorSpace::Deserialize(data.data(), data.size());
    }

    std::span<const uint8_t> dataReference() const
    {
        if (!m_serializedColorSpace)
            m_serializedColorSpace = m_skColorSpace->serialize();
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // Skia port
        if (m_serializedColorSpace)
            return { m_serializedColorSpace->bytes(), m_serializedColorSpace->size() };
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        return { };
    }

private:
    sk_sp<SkColorSpace> m_skColorSpace;
    mutable sk_sp<SkData> m_serializedColorSpace;
};

} // namespace WebKit

#endif // USE(SKIA)
