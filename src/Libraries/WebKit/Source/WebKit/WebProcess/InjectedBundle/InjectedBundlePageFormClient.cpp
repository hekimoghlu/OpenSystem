/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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
#include "InjectedBundlePageFormClient.h"

#include "APIArray.h"
#include "APIDictionary.h"
#include "InjectedBundleNodeHandle.h"
#include "WKAPICast.h"
#include "WKBundleAPICast.h"
#include "WebFrame.h"
#include "WebPage.h"
#include <WebCore/HTMLFormElement.h>
#include <WebCore/HTMLInputElement.h>
#include <WebCore/HTMLTextAreaElement.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InjectedBundlePageFormClient);

InjectedBundlePageFormClient::InjectedBundlePageFormClient(const WKBundlePageFormClientBase* client)
{
    initialize(client);
}

void InjectedBundlePageFormClient::didFocusTextField(WebPage* page, HTMLInputElement& inputElement, WebFrame* frame)
{
    if (!m_client.didFocusTextField)
        return;

    auto nodeHandle = InjectedBundleNodeHandle::getOrCreate(inputElement);
    m_client.didFocusTextField(toAPI(page), toAPI(nodeHandle.get()), toAPI(frame), m_client.base.clientInfo);
}

void InjectedBundlePageFormClient::textFieldDidBeginEditing(WebPage* page, HTMLInputElement& inputElement, WebFrame* frame)
{
    if (!m_client.textFieldDidBeginEditing)
        return;

    RefPtr<InjectedBundleNodeHandle> nodeHandle = InjectedBundleNodeHandle::getOrCreate(inputElement);
    m_client.textFieldDidBeginEditing(toAPI(page), toAPI(nodeHandle.get()), toAPI(frame), m_client.base.clientInfo);
}

void InjectedBundlePageFormClient::textFieldDidEndEditing(WebPage* page, HTMLInputElement& inputElement, WebFrame* frame)
{
    if (!m_client.textFieldDidEndEditing)
        return;

    RefPtr<InjectedBundleNodeHandle> nodeHandle = InjectedBundleNodeHandle::getOrCreate(inputElement);
    m_client.textFieldDidEndEditing(toAPI(page), toAPI(nodeHandle.get()), toAPI(frame), m_client.base.clientInfo);
}

void InjectedBundlePageFormClient::textDidChangeInTextField(WebPage* page, HTMLInputElement& inputElement, WebFrame* frame, bool initiatedByUserTyping)
{
    if (!m_client.textDidChangeInTextField)
        return;

    if (!initiatedByUserTyping)
        return;

    RefPtr<InjectedBundleNodeHandle> nodeHandle = InjectedBundleNodeHandle::getOrCreate(inputElement);
    m_client.textDidChangeInTextField(toAPI(page), toAPI(nodeHandle.get()), toAPI(frame), m_client.base.clientInfo);
}

void InjectedBundlePageFormClient::textDidChangeInTextArea(WebPage* page, HTMLTextAreaElement& textAreaElement, WebFrame* frame)
{
    if (!m_client.textDidChangeInTextArea)
        return;

    RefPtr<InjectedBundleNodeHandle> nodeHandle = InjectedBundleNodeHandle::getOrCreate(textAreaElement);
    m_client.textDidChangeInTextArea(toAPI(page), toAPI(nodeHandle.get()), toAPI(frame), m_client.base.clientInfo);
}

static WKInputFieldActionType toWKInputFieldActionType(API::InjectedBundle::FormClient::InputFieldAction action)
{
    switch (action) {
    case API::InjectedBundle::FormClient::InputFieldAction::MoveUp:
        return WKInputFieldActionTypeMoveUp;
    case API::InjectedBundle::FormClient::InputFieldAction::MoveDown:
        return WKInputFieldActionTypeMoveDown;
    case API::InjectedBundle::FormClient::InputFieldAction::Cancel:
        return WKInputFieldActionTypeCancel;
    case API::InjectedBundle::FormClient::InputFieldAction::InsertTab:
        return WKInputFieldActionTypeInsertTab;
    case API::InjectedBundle::FormClient::InputFieldAction::InsertBacktab:
        return WKInputFieldActionTypeInsertBacktab;
    case API::InjectedBundle::FormClient::InputFieldAction::InsertNewline:
        return WKInputFieldActionTypeInsertNewline;
    case API::InjectedBundle::FormClient::InputFieldAction::InsertDelete:
        return WKInputFieldActionTypeInsertDelete;
    }

    ASSERT_NOT_REACHED();
    return WKInputFieldActionTypeCancel;
}

bool InjectedBundlePageFormClient::shouldPerformActionInTextField(WebPage* page, HTMLInputElement& inputElement, API::InjectedBundle::FormClient::InputFieldAction actionType, WebFrame* frame)
{
    if (!m_client.shouldPerformActionInTextField)
        return false;

    RefPtr<InjectedBundleNodeHandle> nodeHandle = InjectedBundleNodeHandle::getOrCreate(inputElement);
    return m_client.shouldPerformActionInTextField(toAPI(page), toAPI(nodeHandle.get()), toWKInputFieldActionType(actionType), toAPI(frame), m_client.base.clientInfo);
}

void InjectedBundlePageFormClient::willSendSubmitEvent(WebPage* page, HTMLFormElement* formElement, WebFrame* frame, WebFrame* sourceFrame, const Vector<std::pair<String, String>>& values)
{
    if (!m_client.willSendSubmitEvent)
        return;

    RefPtr<InjectedBundleNodeHandle> nodeHandle = InjectedBundleNodeHandle::getOrCreate(formElement);

    API::Dictionary::MapType map;
    for (size_t i = 0; i < values.size(); ++i)
        map.set(values[i].first, API::String::create(values[i].second));
    auto textFieldsMap = API::Dictionary::create(WTFMove(map));

    m_client.willSendSubmitEvent(toAPI(page), toAPI(nodeHandle.get()), toAPI(frame), toAPI(sourceFrame), toAPI(textFieldsMap.ptr()), m_client.base.clientInfo);
}

void InjectedBundlePageFormClient::willSubmitForm(WebPage* page, HTMLFormElement* formElement, WebFrame* frame, WebFrame* sourceFrame, const Vector<std::pair<String, String>>& values, RefPtr<API::Object>& userData)
{
    if (!m_client.willSubmitForm)
        return;

    RefPtr<InjectedBundleNodeHandle> nodeHandle = InjectedBundleNodeHandle::getOrCreate(formElement);

    API::Dictionary::MapType map;
    for (size_t i = 0; i < values.size(); ++i)
        map.set(values[i].first, API::String::create(values[i].second));
    auto textFieldsMap = API::Dictionary::create(WTFMove(map));

    WKTypeRef userDataToPass = 0;
    m_client.willSubmitForm(toAPI(page), toAPI(nodeHandle.get()), toAPI(frame), toAPI(sourceFrame), toAPI(textFieldsMap.ptr()), &userDataToPass, m_client.base.clientInfo);
    userData = adoptRef(toImpl(userDataToPass));
}

void InjectedBundlePageFormClient::didAssociateFormControls(WebPage* page, const Vector<RefPtr<WebCore::Element>>& elements, WebFrame* frame)
{
    if (!m_client.didAssociateFormControls && !m_client.didAssociateFormControlsForFrame)
        return;

    auto elementHandles = elements.map([](auto& element) -> RefPtr<API::Object> {
        return InjectedBundleNodeHandle::getOrCreate(element.get());
    });
    if (!m_client.didAssociateFormControlsForFrame) {
        m_client.didAssociateFormControls(toAPI(page), toAPI(API::Array::create(WTFMove(elementHandles)).ptr()), m_client.base.clientInfo);
        return;
    }

    m_client.didAssociateFormControlsForFrame(toAPI(page), toAPI(API::Array::create(WTFMove(elementHandles)).ptr()), toAPI(frame), m_client.base.clientInfo);
}

bool InjectedBundlePageFormClient::shouldNotifyOnFormChanges(WebPage* page)
{
    if (!m_client.shouldNotifyOnFormChanges)
        return false;

    return m_client.shouldNotifyOnFormChanges(toAPI(page), m_client.base.clientInfo);
}

} // namespace WebKit
