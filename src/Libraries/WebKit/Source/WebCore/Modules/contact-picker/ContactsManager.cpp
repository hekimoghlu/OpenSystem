/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
#include "ContactsManager.h"

#include "Chrome.h"
#include "ContactInfo.h"
#include "ContactProperty.h"
#include "ContactsRequestData.h"
#include "ContactsSelectOptions.h"
#include "Document.h"
#include "JSContactInfo.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalFrame.h"
#include "Navigator.h"
#include "Page.h"
#include "UserGestureIndicator.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ContactsManager);

Ref<ContactsManager> ContactsManager::create(Navigator& navigator)
{
    return adoptRef(*new ContactsManager(navigator));
}

ContactsManager::ContactsManager(Navigator& navigator)
    : m_navigator(navigator)
{
}

ContactsManager::~ContactsManager() = default;


LocalFrame* ContactsManager::frame() const
{
    return m_navigator ? m_navigator->frame() : nullptr;
}

Navigator* ContactsManager::navigator()
{
    return m_navigator.get();
}

void ContactsManager::getProperties(Ref<DeferredPromise>&& promise)
{
    Vector<ContactProperty> properties = { ContactProperty::Email, ContactProperty::Name, ContactProperty::Tel };
    promise->resolve<IDLSequence<IDLEnumeration<ContactProperty>>>(properties);
}

void ContactsManager::select(const Vector<ContactProperty>& properties, const ContactsSelectOptions& options, Ref<DeferredPromise>&& promise)
{
    RefPtr frame = this->frame();
    if (!frame || !frame->isMainFrame() || !frame->document() || !frame->page()) {
        promise->reject(ExceptionCode::InvalidStateError);
        return;
    }

    if (!UserGestureIndicator::processingUserGesture()) {
        promise->reject(ExceptionCode::SecurityError);
        return;
    }

    if (m_contactPickerIsShowing) {
        promise->reject(ExceptionCode::InvalidStateError);
        return;
    }

    if (properties.isEmpty()) {
        promise->reject(ExceptionCode::TypeError);
        return;
    }

    ContactsRequestData requestData;
    requestData.properties = properties;
    requestData.multiple = options.multiple;
    requestData.url = frame->document()->url().truncatedForUseAsBase().string();

    m_contactPickerIsShowing = true;

    frame->page()->chrome().showContactPicker(requestData, [promise = WTFMove(promise), weakThis = WeakPtr { *this }] (std::optional<Vector<ContactInfo>>&& info) {
        if (weakThis)
            weakThis->m_contactPickerIsShowing = false;

        if (info) {
            promise->resolve<IDLSequence<IDLDictionary<ContactInfo>>>(*info);
            return;
        }

        promise->reject(ExceptionCode::UnknownError);
    });
}

} // namespace WebCore
