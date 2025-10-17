/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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

#if PLATFORM(COCOA)

#include <CoreFoundation/CoreFoundation.h>
#include <wtf/RetainPtr.h>
#include <wtf/cf/VectorCF.h>
#include <wtf/cocoa/TypeCastsCocoa.h>

namespace WebKit {

class CoreIPCData {
public:

#ifdef __OBJC__
    CoreIPCData(NSData *nsData)
        : CoreIPCData(bridge_cast(nsData))
    {
    }
#endif

    CoreIPCData(CFDataRef cfData)
        : m_cfData(cfData)
    {
    }

    CoreIPCData(std::optional<std::span<const uint8_t>> data)
        : m_cfData(data ? toCFData(*data) : nullptr)
    {
    }

    RetainPtr<CFDataRef> data() const
    {
        return m_cfData;
    }

    std::optional<std::span<const uint8_t>> dataReference() const
    {
        if (!m_cfData)
            return std::nullopt;
        return span(m_cfData.get());
    }

    RetainPtr<id> toID() const
    {
        return bridge_cast(data());
    }

private:
    RetainPtr<CFDataRef> m_cfData;
};

}

#endif // PLATFORM(COCOA)
