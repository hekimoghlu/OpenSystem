/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 9, 2025.
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
#import "AssertionCapability.h"

#if ENABLE(EXTENSION_CAPABILITIES)

#import "Logging.h"
#import "ProcessLauncher.h"
#import <BrowserEngineKit/BrowserEngineKit.h>

#if USE(LEGACY_EXTENSIONKIT_SPI)
#import "ExtensionKitSoftLink.h"
#endif

namespace WebKit {

AssertionCapability::AssertionCapability(String environmentIdentifier, String domain, String name, Function<void()>&& willInvalidateFunction, Function<void()>&& didInvalidateFunction)
    : m_environmentIdentifier { WTFMove(environmentIdentifier) }
    , m_domain { WTFMove(domain) }
    , m_name { WTFMove(name) }
    , m_willInvalidateBlock { willInvalidateFunction ? makeBlockPtr(WTFMove(willInvalidateFunction)) : nullptr }
    , m_didInvalidateBlock { didInvalidateFunction ? makeBlockPtr(WTFMove(didInvalidateFunction)) : nullptr }
{
    RELEASE_LOG(Process, "AssertionCapability::AssertionCapability: taking assertion %{public}s", m_name.utf8().data());
#if USE(LEGACY_EXTENSIONKIT_SPI)
    if (!ProcessLauncher::hasExtensionsInAppBundle()) {
        _SECapability* capability = [get_SECapabilityClass() assertionWithDomain:m_domain name:m_name environmentIdentifier:m_environmentIdentifier willInvalidate:m_willInvalidateBlock.get() didInvalidate:m_didInvalidateBlock.get()];
        setPlatformCapability(capability);
        return;
    }
#endif
    if (m_name == "Suspended"_s)
        setPlatformCapability([BEProcessCapability suspended]);
    else if (m_name == "Background"_s)
        setPlatformCapability([BEProcessCapability background]);
    else if (m_name == "Foreground"_s)
        setPlatformCapability([BEProcessCapability foreground]);
}

} // namespace WebKit

#endif // ENABLE(EXTENSION_CAPABILITIES)
