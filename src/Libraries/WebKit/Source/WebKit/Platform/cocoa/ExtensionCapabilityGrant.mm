/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#import "config.h"
#import "ExtensionCapabilityGrant.h"

#if ENABLE(EXTENSION_CAPABILITIES)

#import "ExtensionKitSPI.h"
#import "Logging.h"
#import <BrowserEngineKit/BrowserEngineKit.h>
#import <wtf/CrossThreadCopier.h>

namespace WebKit {

static void platformInvalidate(const PlatformGrant& platformGrant)
{
#if USE(LEGACY_EXTENSIONKIT_SPI)
    bool isValid = WTF::switchOn(platformGrant, [&] (auto& grant) {
        return [grant isValid];
    });
    if (!isValid)
        return;

    WTF::switchOn(platformGrant, [&] (const RetainPtr<BEProcessCapabilityGrant>& grant) {
        if (![grant invalidate])
            RELEASE_LOG_ERROR(ProcessCapabilities, "Invalidating grant %{public}@ failed", grant.get());
    }, [&] (const RetainPtr<_SEGrant>& grant) {
        if (![grant invalidateWithError:nil])
            RELEASE_LOG_ERROR(ProcessCapabilities, "Invalidating grant %{public}@ failed", grant.get());
    });
#else
    if (![platformGrant isValid])
        return;
    if (![platformGrant invalidate])
        RELEASE_LOG_ERROR(ProcessCapabilities, "Invalidating grant %{public}@ failed", platformGrant.get());
#endif
}

ExtensionCapabilityGrant::ExtensionCapabilityGrant(String environmentIdentifier)
    : m_environmentIdentifier { WTFMove(environmentIdentifier) }
{
}

ExtensionCapabilityGrant::ExtensionCapabilityGrant(String&& environmentIdentifier, PlatformGrant&& platformGrant)
    : m_environmentIdentifier { WTFMove(environmentIdentifier) }
    , m_platformGrant { WTFMove(platformGrant) }
{
}

ExtensionCapabilityGrant::~ExtensionCapabilityGrant()
{
    setPlatformGrant({ });
}

ExtensionCapabilityGrant ExtensionCapabilityGrant::isolatedCopy() &&
{
    return {
        crossThreadCopy(WTFMove(m_environmentIdentifier)),
        WTFMove(m_platformGrant)
    };
}

bool ExtensionCapabilityGrant::isEmpty() const
{
#if USE(LEGACY_EXTENSIONKIT_SPI)
    return WTF::switchOn(m_platformGrant, [] (auto& grant) {
        return !grant;
    });
#else
    return !m_platformGrant;
#endif
}

bool ExtensionCapabilityGrant::isValid() const
{
#if USE(LEGACY_EXTENSIONKIT_SPI)
    return WTF::switchOn(m_platformGrant, [] (auto& grant) {
        return [grant isValid];
    });
#else
    return [m_platformGrant isValid];
#endif
}

void ExtensionCapabilityGrant::setPlatformGrant(PlatformGrant&& platformGrant)
{
    platformInvalidate(std::exchange(m_platformGrant, WTFMove(platformGrant)));
}

void ExtensionCapabilityGrant::invalidate()
{
    setPlatformGrant({ });
}

} // namespace WebKit

#endif // ENABLE(EXTENSION_CAPABILITIES)
