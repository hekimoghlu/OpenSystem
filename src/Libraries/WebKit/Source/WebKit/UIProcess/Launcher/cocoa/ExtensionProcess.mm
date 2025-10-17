/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
#import "ExtensionProcess.h"

#if USE(EXTENSIONKIT)
#import "AssertionCapability.h"
#import "ExtensionCapability.h"
#import "ExtensionKitSPI.h"
#import <BrowserEngineKit/BrowserEngineKit.h>

#if __has_include(<WebKitAdditions/BEKAdditions.h>)
#import <WebKitAdditions/BEKAdditions.h>
#endif

namespace WebKit {

ExtensionProcess::ExtensionProcess(BEWebContentProcess *process)
    : m_process(process)
{
}

ExtensionProcess::ExtensionProcess(BENetworkingProcess *process)
    : m_process(process)
{
}

ExtensionProcess::ExtensionProcess(BERenderingProcess *process)
    : m_process(process)
{
}

#if USE(LEGACY_EXTENSIONKIT_SPI)
ExtensionProcess::ExtensionProcess(_SEExtensionProcess *process)
    : m_process(process)
{
}
#endif

void ExtensionProcess::invalidate() const
{
    WTF::switchOn(m_process, [&] (auto& process) {
        [process invalidate];
    });
}

OSObjectPtr<xpc_connection_t> ExtensionProcess::makeLibXPCConnection() const
{
    NSError *error = nil;
    OSObjectPtr<xpc_connection_t> xpcConnection;
    WTF::switchOn(m_process, [&] (auto& process) {
        xpcConnection = [process makeLibXPCConnectionError:&error];
    });
    return xpcConnection;
}

PlatformGrant ExtensionProcess::grantCapability(const PlatformCapability& capability, BlockPtr<void()>&& invalidationHandler) const
{
    NSError *error = nil;
    PlatformGrant grant;
#if USE(LEGACY_EXTENSIONKIT_SPI)
    WTF::switchOn(m_process, [&] (auto& process) {
        WTF::switchOn(capability, [&] (const RetainPtr<BEProcessCapability>& capability) {
            grant = [process grantCapability:capability.get() error:&error];
        }, [] (const RetainPtr<_SECapability>&) {
        });
    }, [&] (const RetainPtr<_SEExtensionProcess>& process) {
        WTF::switchOn(capability, [] (const RetainPtr<BEProcessCapability>&) {
        }, [&] (const RetainPtr<_SECapability>& capability) {
            grant = [process grantCapability:capability.get() error:&error];
        });
    });
#else
    WTF::switchOn(m_process, [&] (auto& process) {
#if __has_include(<WebKitAdditions/BEKAdditions.h>)
        GRANT_ADDITIONS
#else
        grant = [process grantCapability:capability.get() error:&error];
#endif
    });
#endif
    return grant;
}

RetainPtr<UIInteraction> ExtensionProcess::createVisibilityPropagationInteraction() const
{
    RetainPtr<UIInteraction> interaction;
    WTF::switchOn(m_process, [&] (RetainPtr<BEWebContentProcess> process) {
        interaction = [process createVisibilityPropagationInteraction];
    }, [&] (RetainPtr<BERenderingProcess> process) {
        interaction = [process createVisibilityPropagationInteraction];
    }, [] (auto& process) {
    });
    return interaction;
}

} // namespace WebKit

#endif
