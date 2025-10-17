/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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

#if ENABLE(EXTENSION_CAPABILITIES)

#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>

OBJC_CLASS BEProcessCapability;
#if USE(LEGACY_EXTENSIONKIT_SPI)
OBJC_CLASS _SECapability;
#endif

namespace WebKit {

#if USE(LEGACY_EXTENSIONKIT_SPI)
using PlatformCapability = std::variant<RetainPtr<BEProcessCapability>, RetainPtr<_SECapability>>;
#else
using PlatformCapability = RetainPtr<BEProcessCapability>;
#endif

class ExtensionCapability {
public:
    virtual ~ExtensionCapability() = default;
    virtual String environmentIdentifier() const = 0;
    const PlatformCapability& platformCapability() const { return m_platformCapability; }

    bool hasPlatformCapability() const { return platformCapabilityIsValid(m_platformCapability); }

    static bool platformCapabilityIsValid(const PlatformCapability& platformCapability)
    {
#if USE(LEGACY_EXTENSIONKIT_SPI)
        return WTF::switchOn(platformCapability, [] (auto& capability) {
            return !!capability;
        });
#else
        return !!platformCapability;
#endif
    }

protected:
    ExtensionCapability() = default;
    void setPlatformCapability(PlatformCapability&& capability) { m_platformCapability = WTFMove(capability); }

private:
    PlatformCapability m_platformCapability;
};

} // namespace WebKit

#endif // ENABLE(EXTENSION_CAPABILITIES)
