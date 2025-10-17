/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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

#include <WebCore/IntRect.h>
#include <WebCore/ValidationMessageClient.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Element;
class WeakPtrImplWithEventTargetData;

}

namespace WebKit {

class WebPage;

class WebValidationMessageClient final : public WebCore::ValidationMessageClient {
    WTF_MAKE_TZONE_ALLOCATED(WebValidationMessageClient);
public:
    explicit WebValidationMessageClient(WebPage&);
    ~WebValidationMessageClient();

    // ValidationMessageClient API.
    void documentDetached(WebCore::Document&) final;
    void showValidationMessage(const WebCore::Element& anchor, const String& message) final;
    void hideValidationMessage(const WebCore::Element& anchor) final;
    void hideAnyValidationMessage() final;
    bool isValidationMessageVisible(const WebCore::Element& anchor) final;
    void updateValidationBubbleStateIfNeeded() final;

private:
    WeakPtr<WebPage> m_page;
    WeakPtr<const WebCore::Element, WebCore::WeakPtrImplWithEventTargetData> m_currentAnchor;
    WebCore::IntRect m_currentAnchorRect;
};

}
