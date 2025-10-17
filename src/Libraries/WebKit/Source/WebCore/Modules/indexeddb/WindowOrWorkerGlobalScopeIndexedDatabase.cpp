/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#include "WindowOrWorkerGlobalScopeIndexedDatabase.h"

#include "Document.h"
#include "IDBConnectionProxy.h"
#include "IDBFactory.h"
#include "LocalDOMWindow.h"
#include "LocalDOMWindowProperty.h"
#include "Page.h"
#include "Supplementable.h"
#include "WorkerGlobalScope.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class DOMWindowIndexedDatabase : public LocalDOMWindowProperty, public Supplement<LocalDOMWindow> {
    WTF_MAKE_TZONE_ALLOCATED(DOMWindowIndexedDatabase);
public:
    explicit DOMWindowIndexedDatabase(LocalDOMWindow&);
    virtual ~DOMWindowIndexedDatabase() = default;

    static DOMWindowIndexedDatabase* from(LocalDOMWindow&);
    IDBFactory* indexedDB();

private:
    static ASCIILiteral supplementName() { return "DOMWindowIndexedDatabase"_s; }

    RefPtr<IDBFactory> m_idbFactory;
};

class WorkerGlobalScopeIndexedDatabase : public Supplement<WorkerGlobalScope> {
    WTF_MAKE_TZONE_ALLOCATED(WorkerGlobalScopeIndexedDatabase);
public:
    explicit WorkerGlobalScopeIndexedDatabase(IDBClient::IDBConnectionProxy&);
    virtual ~WorkerGlobalScopeIndexedDatabase() = default;

    static WorkerGlobalScopeIndexedDatabase* from(WorkerGlobalScope&);
    IDBFactory* indexedDB();

private:
    static ASCIILiteral supplementName() { return "WorkerGlobalScopeIndexedDatabase"_s; }

    RefPtr<IDBFactory> m_idbFactory;
    Ref<IDBClient::IDBConnectionProxy> m_connectionProxy;
};

// DOMWindowIndexedDatabase supplement.

WTF_MAKE_TZONE_ALLOCATED_IMPL(DOMWindowIndexedDatabase);

DOMWindowIndexedDatabase::DOMWindowIndexedDatabase(LocalDOMWindow& window)
    : LocalDOMWindowProperty(&window)
{
}

DOMWindowIndexedDatabase* DOMWindowIndexedDatabase::from(LocalDOMWindow& window)
{
    auto* supplement = static_cast<DOMWindowIndexedDatabase*>(Supplement<LocalDOMWindow>::from(&window, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<DOMWindowIndexedDatabase>(window);
        supplement = newSupplement.get();
        provideTo(&window, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

IDBFactory* DOMWindowIndexedDatabase::indexedDB()
{
    RefPtr window = this->window();
    if (!window)
        return nullptr;

    RefPtr document = window->document();
    if (!document)
        return nullptr;

    auto* page = document->page();
    if (!page)
        return nullptr;

    if (!window->isCurrentlyDisplayedInFrame())
        return nullptr;

    if (!m_idbFactory) {
        auto* connectionProxy = document->idbConnectionProxy();
        if (!connectionProxy)
            return nullptr;

        m_idbFactory = IDBFactory::create(*connectionProxy);
    }

    return m_idbFactory.get();
}

// WorkerGlobalScope supplement.

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerGlobalScopeIndexedDatabase);

WorkerGlobalScopeIndexedDatabase::WorkerGlobalScopeIndexedDatabase(IDBClient::IDBConnectionProxy& connectionProxy)
    : m_connectionProxy(connectionProxy)
{
}

WorkerGlobalScopeIndexedDatabase* WorkerGlobalScopeIndexedDatabase::from(WorkerGlobalScope& scope)
{
    auto* supplement = static_cast<WorkerGlobalScopeIndexedDatabase*>(Supplement<WorkerGlobalScope>::from(&scope, supplementName()));
    if (!supplement) {
        auto* connectionProxy = scope.idbConnectionProxy();
        if (!connectionProxy)
            return nullptr;

        auto newSupplement = makeUnique<WorkerGlobalScopeIndexedDatabase>(*connectionProxy);
        supplement = newSupplement.get();
        provideTo(&scope, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

IDBFactory* WorkerGlobalScopeIndexedDatabase::indexedDB()
{
    if (!m_idbFactory)
        m_idbFactory = IDBFactory::create(m_connectionProxy.get());
    return m_idbFactory.get();
}


IDBFactory* WindowOrWorkerGlobalScopeIndexedDatabase::indexedDB(WorkerGlobalScope& scope)
{
    auto* scopeIDB = WorkerGlobalScopeIndexedDatabase::from(scope);
    return scopeIDB ? scopeIDB->indexedDB() : nullptr;
}

IDBFactory* WindowOrWorkerGlobalScopeIndexedDatabase::indexedDB(DOMWindow& window)
{
    RefPtr localWindow = dynamicDowncast<LocalDOMWindow>(window);
    if (!localWindow)
        return nullptr;
    return DOMWindowIndexedDatabase::from(*localWindow)->indexedDB();
}

} // namespace WebCore
