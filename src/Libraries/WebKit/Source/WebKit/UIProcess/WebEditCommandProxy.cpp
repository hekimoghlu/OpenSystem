/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
#include "WebEditCommandProxy.h"

#include "MessageSenderInlines.h"
#include "UndoOrRedo.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"
#include <WebCore/LocalizedStrings.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
using namespace WebCore;

WebEditCommandProxy::WebEditCommandProxy(WebUndoStepID commandID, const String& label, WebPageProxy& page)
    : m_commandID(commandID)
    , m_label(label)
    , m_page(page)
{
    m_page->addEditCommand(*this);
}

WebEditCommandProxy::~WebEditCommandProxy()
{
    if (m_page)
        m_page->removeEditCommand(*this);
}

void WebEditCommandProxy::unapply()
{
    if (!m_page || !m_page->hasRunningProcess())
        return;

    m_page->legacyMainFrameProcess().send(Messages::WebPage::UnapplyEditCommand(m_commandID), m_page->webPageIDInMainFrameProcess(), IPC::SendOption::DispatchMessageEvenWhenWaitingForSyncReply);
    m_page->registerEditCommand(*this, UndoOrRedo::Redo);
}

void WebEditCommandProxy::reapply()
{
    if (!m_page || !m_page->hasRunningProcess())
        return;

    m_page->legacyMainFrameProcess().send(Messages::WebPage::ReapplyEditCommand(m_commandID), m_page->webPageIDInMainFrameProcess(), IPC::SendOption::DispatchMessageEvenWhenWaitingForSyncReply);
    m_page->registerEditCommand(*this, UndoOrRedo::Undo);
}

} // namespace WebKit
