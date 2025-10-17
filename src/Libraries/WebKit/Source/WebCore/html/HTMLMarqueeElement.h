/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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

#include "ActiveDOMObject.h"
#include "HTMLElement.h"

namespace WebCore {

class RenderMarquee;

class HTMLMarqueeElement final : public HTMLElement, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HTMLMarqueeElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLMarqueeElement);
public:
    static Ref<HTMLMarqueeElement> create(const QualifiedName&, Document&);

    // ActiveDOMObject.
    void ref() const final { HTMLElement::ref(); }
    void deref() const final { HTMLElement::deref(); }

    int minimumDelay() const;

    WEBCORE_EXPORT void start();
    WEBCORE_EXPORT void stop() final;
    
    // Number of pixels to move on each scroll movement. Defaults to 6.
    WEBCORE_EXPORT unsigned scrollAmount() const;
    WEBCORE_EXPORT void setScrollAmount(unsigned);
    
    // Interval between each scroll movement, in milliseconds. Defaults to 60.
    WEBCORE_EXPORT unsigned scrollDelay() const;
    WEBCORE_EXPORT void setScrollDelay(unsigned);
    
    // Loop count. -1 means loop indefinitely.
    WEBCORE_EXPORT int loop() const;
    WEBCORE_EXPORT ExceptionOr<void> setLoop(int);
    
private:
    HTMLMarqueeElement(const QualifiedName&, Document&);

    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;

    // ActiveDOMObject.
    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) final;
    void suspend(ReasonForSuspension) final;
    void resume() final;

    RenderMarquee* renderMarquee() const;
};

} // namespace WebCore
