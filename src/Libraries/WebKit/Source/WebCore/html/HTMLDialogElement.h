/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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

#include "HTMLElement.h"

namespace WebCore {

class HTMLDialogElement final : public HTMLElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLDialogElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLDialogElement);
public:
    template<typename... Args> static Ref<HTMLDialogElement> create(Args&&... args) { return adoptRef(*new HTMLDialogElement(std::forward<Args>(args)...)); }

    bool isOpen() const { return hasAttribute(HTMLNames::openAttr); }

    const String& returnValue() const { return m_returnValue; }
    void setReturnValue(String&& value) { m_returnValue = WTFMove(value); }

    ExceptionOr<void> show();
    ExceptionOr<void> showModal();
    void close(const String&);
    void requestClose(const String&);

    bool isModal() const { return m_isModal; };

    void queueCancelTask();

    void runFocusingSteps();

    bool isValidCommandType(const CommandType) final;
    bool handleCommandInternal(const HTMLFormControlElement& invoker, const CommandType&) final;

private:
    HTMLDialogElement(const QualifiedName&, Document&);

    void removedFromAncestor(RemovalType, ContainerNode& oldParentOfRemovedTree) final;
    void setIsModal(bool newValue);
    bool supportsFocus() const final;

    String m_returnValue;
    bool m_isModal { false };
    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_previouslyFocusedElement;
};

} // namespace WebCore
