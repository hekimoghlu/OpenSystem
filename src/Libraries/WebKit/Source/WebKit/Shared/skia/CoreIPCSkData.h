/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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

#include <skia/core/SkData.h>

namespace WebKit {

class CoreIPCSkData {
public:
    CoreIPCSkData(sk_sp<SkData> skData)
        : m_skData(WTFMove(skData))
    {
    }

    static sk_sp<SkData> create(std::span<const uint8_t> data)
    {
        return SkData::MakeWithCopy(data.data(), data.size());
    }

    std::span<const uint8_t> dataReference() const
    {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // Skia port
        return { m_skData->bytes(), m_skData->size() };
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }

private:
    sk_sp<SkData> m_skData;
};

} // namespace WebKit

#endif // USE(SKIA)
