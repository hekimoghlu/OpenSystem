/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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

#include "ExtensionCapability.h"
#include "ExtensionCapabilityGrant.h"

#include <wtf/BlockPtr.h>
#include <wtf/OSObjectPtr.h>
#include <wtf/RetainPtr.h>

#if USE(EXTENSIONKIT)
OBJC_CLASS BEWebContentProcess;
OBJC_CLASS BENetworkingProcess;
OBJC_CLASS BERenderingProcess;
OBJC_CLASS BEProcessCapability;
OBJC_CLASS _SEExtensionProcess;
OBJC_PROTOCOL(BEProcessCapabilityGrant);
OBJC_PROTOCOL(UIInteraction);

namespace WebKit {

#if USE(LEGACY_EXTENSIONKIT_SPI)
using ExtensionProcessVariant = std::variant<RetainPtr<BEWebContentProcess>, RetainPtr<BENetworkingProcess>, RetainPtr<BERenderingProcess>, RetainPtr<_SEExtensionProcess>>;
#else
using ExtensionProcessVariant = std::variant<RetainPtr<BEWebContentProcess>, RetainPtr<BENetworkingProcess>, RetainPtr<BERenderingProcess>>;
#endif

class ExtensionProcess {
public:
    ExtensionProcess(BEWebContentProcess *);
    ExtensionProcess(BENetworkingProcess *);
    ExtensionProcess(BERenderingProcess *);
#if USE(LEGACY_EXTENSIONKIT_SPI)
    ExtensionProcess(_SEExtensionProcess *);
#endif

    void invalidate() const;
    OSObjectPtr<xpc_connection_t> makeLibXPCConnection() const;
    PlatformGrant grantCapability(const PlatformCapability&, BlockPtr<void()>&& invalidationHandler = ^{ }) const;
    RetainPtr<UIInteraction> createVisibilityPropagationInteraction() const;

private:
    ExtensionProcessVariant m_process;
};

} // namespace WebKit

#endif // USE(EXTENSIONKIT)
