/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#include "InspectorDOMStorageAgent.h"

#include "DOMException.h"
#include "Database.h"
#include "Document.h"
#include "InspectorPageAgent.h"
#include "InstrumentingAgents.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "Page.h"
#include "SecurityOrigin.h"
#include "SecurityOriginData.h"
#include "Storage.h"
#include "StorageNamespace.h"
#include "StorageNamespaceProvider.h"
#include "StorageType.h"
#include "VoidCallback.h"
#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <wtf/JSONValues.h>
#include <wtf/TZoneMallocInlines.h>


namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorDOMStorageAgent);

InspectorDOMStorageAgent::InspectorDOMStorageAgent(PageAgentContext& context)
    : InspectorAgentBase("DOMStorage"_s, context)
    , m_frontendDispatcher(makeUnique<Inspector::DOMStorageFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(Inspector::DOMStorageBackendDispatcher::create(context.backendDispatcher, this))
    , m_inspectedPage(context.inspectedPage)
{
}

InspectorDOMStorageAgent::~InspectorDOMStorageAgent() = default;

void InspectorDOMStorageAgent::didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*)
{
}

void InspectorDOMStorageAgent::willDestroyFrontendAndBackend(Inspector::DisconnectReason)
{
    disable();
}

Inspector::Protocol::ErrorStringOr<void> InspectorDOMStorageAgent::enable()
{
    if (m_instrumentingAgents.enabledDOMStorageAgent() == this)
        return makeUnexpected("DOMStorage domain already enabled"_s);

    m_instrumentingAgents.setEnabledDOMStorageAgent(this);

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorDOMStorageAgent::disable()
{
    if (m_instrumentingAgents.enabledDOMStorageAgent() != this)
        return makeUnexpected("DOMStorage domain already disabled"_s);

    m_instrumentingAgents.setEnabledDOMStorageAgent(nullptr);

    return { };
}

Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::DOMStorage::Item>>> InspectorDOMStorageAgent::getDOMStorageItems(Ref<JSON::Object>&& storageId)
{
    Inspector::Protocol::ErrorString errorString;

    LocalFrame* frame;
    RefPtr<StorageArea> storageArea = findStorageArea(errorString, WTFMove(storageId), frame);
    if (!storageArea)
        return makeUnexpected(errorString);

    auto storageItems = JSON::ArrayOf<JSON::ArrayOf<String>>::create();
    for (unsigned i = 0; i < storageArea->length(); ++i) {
        String key = storageArea->key(i);
        String value = storageArea->item(key);

        auto entry = JSON::ArrayOf<String>::create();
        entry->addItem(key);
        entry->addItem(value);
        storageItems->addItem(WTFMove(entry));
    }
    return storageItems;
}

Inspector::Protocol::ErrorStringOr<void> InspectorDOMStorageAgent::setDOMStorageItem(Ref<JSON::Object>&& storageId, const String& key, const String& value)
{
    Inspector::Protocol::ErrorString errorString;

    LocalFrame* frame;
    RefPtr<StorageArea> storageArea = findStorageArea(errorString, WTFMove(storageId), frame);
    if (!storageArea)
        return makeUnexpected(errorString);

    bool quotaException = false;
    storageArea->setItem(*frame, key, value, quotaException);
    if (quotaException)
        return makeUnexpected(DOMException::name(ExceptionCode::QuotaExceededError));

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorDOMStorageAgent::removeDOMStorageItem(Ref<JSON::Object>&& storageId, const String& key)
{
    Inspector::Protocol::ErrorString errorString;

    LocalFrame* frame;
    RefPtr<StorageArea> storageArea = findStorageArea(errorString, WTFMove(storageId), frame);
    if (!storageArea)
        return makeUnexpected(errorString);

    storageArea->removeItem(*frame, key);

    return { };
}

Inspector::Protocol::ErrorStringOr<void> InspectorDOMStorageAgent::clearDOMStorageItems(Ref<JSON::Object>&& storageId)
{
    Inspector::Protocol::ErrorString errorString;

    LocalFrame* frame;
    auto storageArea = findStorageArea(errorString, WTFMove(storageId), frame);
    if (!storageArea)
        return makeUnexpected(errorString);

    storageArea->clear(*frame);

    return { };
}

String InspectorDOMStorageAgent::storageId(Storage& storage)
{
    auto* document = storage.frame()->document();
    ASSERT(document);
    auto* window = document->domWindow();
    ASSERT(window);
    Ref<SecurityOrigin> securityOrigin = document->securityOrigin();
    bool isLocalStorage = window->optionalLocalStorage() == &storage;
    return InspectorDOMStorageAgent::storageId(securityOrigin, isLocalStorage)->toJSONString();
}

Ref<Inspector::Protocol::DOMStorage::StorageId> InspectorDOMStorageAgent::storageId(const SecurityOrigin& securityOrigin, bool isLocalStorage)
{
    return Inspector::Protocol::DOMStorage::StorageId::create()
        .setSecurityOrigin(securityOrigin.toRawString())
        .setIsLocalStorage(isLocalStorage)
        .release();
}

void InspectorDOMStorageAgent::didDispatchDOMStorageEvent(const String& key, const String& oldValue, const String& newValue, StorageType storageType, const SecurityOrigin& securityOrigin)
{
    auto id = InspectorDOMStorageAgent::storageId(securityOrigin, storageType == StorageType::Local);

    if (key.isNull())
        m_frontendDispatcher->domStorageItemsCleared(WTFMove(id));
    else if (newValue.isNull())
        m_frontendDispatcher->domStorageItemRemoved(WTFMove(id), key);
    else if (oldValue.isNull())
        m_frontendDispatcher->domStorageItemAdded(WTFMove(id), key, newValue);
    else
        m_frontendDispatcher->domStorageItemUpdated(WTFMove(id), key, oldValue, newValue);
}

RefPtr<StorageArea> InspectorDOMStorageAgent::findStorageArea(Inspector::Protocol::ErrorString& errorString, Ref<JSON::Object>&& storageId, LocalFrame*& targetFrame)
{
    auto securityOrigin = storageId->getString("securityOrigin"_s);
    if (!securityOrigin) {
        errorString = "Missing securityOrigin in given storageId"_s;
        return nullptr;
    }

    auto isLocalStorage = storageId->getBoolean("isLocalStorage"_s);
    if (!isLocalStorage) {
        errorString = "Missing isLocalStorage in given storageId"_s;
        return nullptr;
    }

    targetFrame = InspectorPageAgent::findFrameWithSecurityOrigin(m_inspectedPage, securityOrigin);
    if (!targetFrame) {
        errorString = "Missing frame for given securityOrigin"_s;
        return nullptr;
    }

    auto& document = *targetFrame->document();
    if (!*isLocalStorage)
        return m_inspectedPage->storageNamespaceProvider().sessionStorageArea(document);
    return m_inspectedPage->storageNamespaceProvider().localStorageArea(document);
}

} // namespace WebCore
