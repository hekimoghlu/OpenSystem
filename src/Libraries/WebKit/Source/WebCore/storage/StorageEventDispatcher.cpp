/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#include "StorageEventDispatcher.h"

#include "Document.h"
#include "EventNames.h"
#include "InspectorInstrumentation.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PageGroup.h"
#include "SecurityOrigin.h"
#include "SecurityOriginData.h"
#include "StorageEvent.h"
#include "StorageType.h"

namespace WebCore {

template<StorageType storageType>
static void dispatchStorageEvents(const String& key, const String& oldValue, const String& newValue, const SecurityOrigin& securityOrigin, const String& url, const Function<bool(Storage&)>& isSourceStorage, const Function<bool(Page&)>& isRelevantPage)
{
    Vector<Ref<LocalDOMWindow>> windows;
    LocalDOMWindow::forEachWindowInterestedInStorageEvents([&](auto& window) {
        auto storage = isLocalStorage(storageType) ? window.optionalLocalStorage() : window.optionalSessionStorage();
        if (!storage)
            return;
        // Send events only to our page.
        if (RefPtr page = window.page(); !page || !isRelevantPage(*page))
            return;
        if (isSourceStorage(*storage))
            return;
        if (!securityOrigin.equal(*window.securityOrigin()))
            return;
        windows.append(window);
    });

    for (auto& window : windows) {
        RefPtr document = window->document();
        auto result = isLocalStorage(storageType) ? window->localStorage() : window->sessionStorage();
        if (!result.hasException()) // https://html.spec.whatwg.org/multipage/webstorage.html#the-storage-event:event-storage
            document->queueTaskToDispatchEventOnWindow(TaskSource::DOMManipulation, StorageEvent::create(eventNames().storageEvent, key, oldValue, newValue, url, result.releaseReturnValue()));
    }
}

void StorageEventDispatcher::dispatchSessionStorageEvents(const String& key, const String& oldValue, const String& newValue, Page& page, const SecurityOrigin& securityOrigin, const String& url, const Function<bool(Storage&)>& isSourceStorage)
{
    InspectorInstrumentation::didDispatchDOMStorageEvent(page, key, oldValue, newValue, StorageType::Session, securityOrigin);
    dispatchStorageEvents<StorageType::Session>(key, oldValue, newValue, securityOrigin, url, isSourceStorage, [&](auto& windowPage) {
        return &windowPage == &page;
    });
}

void StorageEventDispatcher::dispatchLocalStorageEvents(const String& key, const String& oldValue, const String& newValue, PageGroup* pageGroup, const SecurityOrigin& securityOrigin, const String& url, const Function<bool(Storage&)>& isSourceStorage)
{
    if (!pageGroup) {
        Page::forEachPage([&](Page& page) {
            InspectorInstrumentation::didDispatchDOMStorageEvent(page, key, oldValue, newValue, StorageType::Local, securityOrigin);
        });
        dispatchStorageEvents<StorageType::Local>(key, oldValue, newValue, securityOrigin, url, isSourceStorage, [](auto&) { return true; });
        return;
    }

    auto& pagesInGroup = pageGroup->pages();
    for (auto& page : pagesInGroup)
        InspectorInstrumentation::didDispatchDOMStorageEvent(page, key, oldValue, newValue, StorageType::Local, securityOrigin);
    dispatchStorageEvents<StorageType::Local>(key, oldValue, newValue, securityOrigin, url, isSourceStorage, [&](auto& page) {
        return pagesInGroup.contains(page);
    });
}

} // namespace WebCore
