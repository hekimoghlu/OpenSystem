/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
#include "config.h"
#include "NavigatorBase.h"

#include "Document.h"
#include "GPU.h"
#include "ScriptTelemetryCategory.h"
#include "ServiceWorkerContainer.h"
#include "StorageManager.h"
#include "WebCoreOpaqueRoot.h"
#include "WebLockManager.h"
#include <mutex>
#include <wtf/Language.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/NumberOfCores.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakRandom.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

#if OS(LINUX)
#include "sys/utsname.h"
#include <wtf/StdLibExtras.h>
#endif

#if PLATFORM(IOS_FAMILY)
#import <pal/system/ios/Device.h>
#endif

#ifndef WEBCORE_NAVIGATOR_PRODUCT
#define WEBCORE_NAVIGATOR_PRODUCT "Gecko"_s
#endif // ifndef WEBCORE_NAVIGATOR_PRODUCT

#ifndef WEBCORE_NAVIGATOR_PRODUCT_SUB
#define WEBCORE_NAVIGATOR_PRODUCT_SUB "20030107"_s
#endif // ifndef WEBCORE_NAVIGATOR_PRODUCT_SUB

#ifndef WEBCORE_NAVIGATOR_VENDOR
#define WEBCORE_NAVIGATOR_VENDOR "Apple Computer, Inc."_s
#endif // ifndef WEBCORE_NAVIGATOR_VENDOR

#ifndef WEBCORE_NAVIGATOR_VENDOR_SUB
#define WEBCORE_NAVIGATOR_VENDOR_SUB emptyString()
#endif // ifndef WEBCORE_NAVIGATOR_VENDOR_SUB

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(NavigatorBase);

NavigatorBase::NavigatorBase(ScriptExecutionContext* context)
    : ContextDestructionObserver(context)
{
}

NavigatorBase::~NavigatorBase() = default;

String NavigatorBase::appName()
{
    return "Netscape"_s;
}

String NavigatorBase::appVersion() const
{
    // Version is everything in the user agent string past the "Mozilla/" prefix.
    const String& agent = userAgent();
    return agent.substring(agent.find('/') + 1);
}

String NavigatorBase::platform() const
{
#if OS(LINUX)
    static LazyNeverDestroyed<String> platformName;
    static std::once_flag onceKey;
    std::call_once(onceKey, [] {
        struct utsname osname;
        platformName.construct(uname(&osname) >= 0 ? makeString(unsafeSpan(osname.sysname), " "_s, unsafeSpan(osname.machine)) : emptyString());
    });
    return platformName->isolatedCopy();
#elif PLATFORM(IOS_FAMILY)
    return PAL::deviceName();
#elif OS(MACOS)
    return "MacIntel"_s;
#elif OS(WINDOWS)
    return "Win32"_s;
#else
    return ""_s;
#endif
}

String NavigatorBase::appCodeName()
{
    return "Mozilla"_s;
}

String NavigatorBase::product()
{
    return WEBCORE_NAVIGATOR_PRODUCT;
}

String NavigatorBase::productSub()
{
    return WEBCORE_NAVIGATOR_PRODUCT_SUB;
}

String NavigatorBase::vendor()
{
    return WEBCORE_NAVIGATOR_VENDOR;
}

String NavigatorBase::vendorSub()
{
    return WEBCORE_NAVIGATOR_VENDOR_SUB;
}

String NavigatorBase::language()
{
    return defaultLanguage();
}

Vector<String> NavigatorBase::languages()
{
    // We intentionally expose only the primary language for privacy reasons.
    return { defaultLanguage() };
}

StorageManager& NavigatorBase::storage()
{
    if (!m_storageManager)
        m_storageManager = StorageManager::create(*this);

    return *m_storageManager;
}

WebLockManager& NavigatorBase::locks()
{
    if (!m_webLockManager)
        m_webLockManager = WebLockManager::create(*this);

    return *m_webLockManager;
}

ServiceWorkerContainer& NavigatorBase::serviceWorker()
{
    ASSERT(!scriptExecutionContext() || scriptExecutionContext()->settingsValues().serviceWorkersEnabled);
    if (!m_serviceWorkerContainer)
        m_serviceWorkerContainer = ServiceWorkerContainer::create(protectedScriptExecutionContext().get(), *this).moveToUniquePtr();
    return *m_serviceWorkerContainer;
}

ExceptionOr<ServiceWorkerContainer&> NavigatorBase::serviceWorker(ScriptExecutionContext& context)
{
    if (RefPtr document = dynamicDowncast<Document>(context); document && document->isSandboxed(SandboxFlag::Origin))
        return Exception { ExceptionCode::SecurityError, "Service Worker is disabled because the context is sandboxed and lacks the 'allow-same-origin' flag"_s };
    return serviceWorker();
}

int NavigatorBase::hardwareConcurrency(ScriptExecutionContext& context)
{
    static int numberOfCores;

    if (context.requiresScriptExecutionTelemetry(ScriptTelemetryCategory::HardwareConcurrency)) {
        auto randomSeed = static_cast<unsigned>(context.noiseInjectionHashSalt().value_or(0));
        return 1 + WeakRandom { randomSeed }.getUint32(63);
    }

    static std::once_flag once;
    std::call_once(once, [] {
        // Enforce a maximum for the number of cores reported to mitigate
        // fingerprinting for the minority of machines with large numbers of cores.
        // If machines with more than 8 cores become commonplace, we should bump this number.
        // see https://bugs.webkit.org/show_bug.cgi?id=132588 for the
        // rationale behind this decision.
        if (WTF::numberOfProcessorCores() < 8)
            numberOfCores = 4;
        else
            numberOfCores = 8;
    });

    return numberOfCores;
}

WebCoreOpaqueRoot root(NavigatorBase* navigator)
{
    return WebCoreOpaqueRoot { navigator };
}

} // namespace WebCore
