/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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

#include "Timer.h"
#include <memory>
#include <wtf/Noncopyable.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class WeakPtrImplWithEventTargetData;
class HTMLElement;
class Node;
class ValidationMessageClient;

// FIXME: We should remove the code for !validationMessageClient() when all
// ports supporting interactive validation switch to ValidationMessageClient.
class ValidationMessage : public RefCountedAndCanMakeWeakPtr<ValidationMessage> {
    WTF_MAKE_TZONE_ALLOCATED(ValidationMessage);
    WTF_MAKE_NONCOPYABLE(ValidationMessage);
public:
    static Ref<ValidationMessage> create(HTMLElement&);
    ~ValidationMessage();

    void updateValidationMessage(HTMLElement&, const String&);
    void requestToHideMessage();
    bool isVisible() const;
    bool shadowTreeContains(const Node&) const;
    void adjustBubblePosition();

private:
    explicit ValidationMessage(HTMLElement&);

    ValidationMessageClient* validationMessageClient() const;
    void setMessage(const String&);
    void setMessageDOMAndStartTimer();
    void buildBubbleTree();
    void deleteBubbleTree();

    WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData> m_element;
    String m_message;
    std::unique_ptr<Timer> m_timer;
    RefPtr<HTMLElement> m_bubble;
    RefPtr<HTMLElement> m_messageHeading;
    RefPtr<HTMLElement> m_messageBody;
};

} // namespace WebCore
