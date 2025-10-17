/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#include "WindowOrWorkerGlobalScopeCaches.h"

#include "CacheStorageProvider.h"
#include "DOMCacheStorage.h"
#include "Document.h"
#include "LocalDOMWindow.h"
#include "LocalDOMWindowProperty.h"
#include "LocalFrame.h"
#include "Page.h"
#include "Supplementable.h"
#include "WorkerGlobalScope.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class DOMWindowCaches : public Supplement<LocalDOMWindow>, public LocalDOMWindowProperty {
    WTF_MAKE_TZONE_ALLOCATED(DOMWindowCaches);
public:
    explicit DOMWindowCaches(LocalDOMWindow&);
    virtual ~DOMWindowCaches() = default;

    static DOMWindowCaches* from(LocalDOMWindow&);
    DOMCacheStorage* caches() const;

private:
    static ASCIILiteral supplementName() { return "DOMWindowCaches"_s; }

    mutable RefPtr<DOMCacheStorage> m_caches;
};

class WorkerGlobalScopeCaches : public Supplement<WorkerGlobalScope> {
    WTF_MAKE_TZONE_ALLOCATED(WorkerGlobalScopeCaches);
public:
    explicit WorkerGlobalScopeCaches(WorkerGlobalScope&);
    virtual ~WorkerGlobalScopeCaches() = default;

    static WorkerGlobalScopeCaches* from(WorkerGlobalScope&);
    DOMCacheStorage* caches() const;

private:
    static ASCIILiteral supplementName() { return "WorkerGlobalScopeCaches"_s; }

    WeakRef<WorkerGlobalScope, WeakPtrImplWithEventTargetData> m_scope;
    mutable RefPtr<DOMCacheStorage> m_caches;
};

// DOMWindowCaches supplement.

WTF_MAKE_TZONE_ALLOCATED_IMPL(DOMWindowCaches);

DOMWindowCaches::DOMWindowCaches(LocalDOMWindow& window)
    : LocalDOMWindowProperty(&window)
{
}

DOMWindowCaches* DOMWindowCaches::from(LocalDOMWindow& window)
{
    auto* supplement = static_cast<DOMWindowCaches*>(Supplement<LocalDOMWindow>::from(&window, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<DOMWindowCaches>(window);
        supplement = newSupplement.get();
        provideTo(&window, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

DOMCacheStorage* DOMWindowCaches::caches() const
{
    ASSERT(frame());
    ASSERT(frame()->document());
    if (!m_caches && frame()->page())
        m_caches = DOMCacheStorage::create(*frame()->document(), frame()->page()->cacheStorageProvider().createCacheStorageConnection());
    return m_caches.get();
}

// WorkerGlobalScope supplement.

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerGlobalScopeCaches);

WorkerGlobalScopeCaches::WorkerGlobalScopeCaches(WorkerGlobalScope& scope)
    : m_scope(scope)
{
}

WorkerGlobalScopeCaches* WorkerGlobalScopeCaches::from(WorkerGlobalScope& scope)
{
    auto* supplement = static_cast<WorkerGlobalScopeCaches*>(Supplement<WorkerGlobalScope>::from(&scope, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<WorkerGlobalScopeCaches>(scope);
        supplement = newSupplement.get();
        provideTo(&scope, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

DOMCacheStorage* WorkerGlobalScopeCaches::caches() const
{
    if (!m_caches) {
        Ref scope = m_scope.get();
        m_caches = DOMCacheStorage::create(scope, scope->cacheStorageConnection());
    }
    return m_caches.get();
}

// WindowOrWorkerGlobalScopeCaches.

ExceptionOr<DOMCacheStorage*> WindowOrWorkerGlobalScopeCaches::caches(ScriptExecutionContext& context, DOMWindow& window)
{
    if (downcast<Document>(context).isSandboxed(SandboxFlag::Origin))
        return Exception { ExceptionCode::SecurityError, "Cache storage is disabled because the context is sandboxed and lacks the 'allow-same-origin' flag"_s };

    RefPtr localWindow = dynamicDowncast<LocalDOMWindow>(window);
    if (!localWindow)
        return nullptr;

    if (!localWindow->isCurrentlyDisplayedInFrame())
        return nullptr;

    return DOMWindowCaches::from(*localWindow)->caches();
}

DOMCacheStorage* WindowOrWorkerGlobalScopeCaches::caches(ScriptExecutionContext&, WorkerGlobalScope& scope)
{
    return WorkerGlobalScopeCaches::from(scope)->caches();
}

} // namespace WebCore
