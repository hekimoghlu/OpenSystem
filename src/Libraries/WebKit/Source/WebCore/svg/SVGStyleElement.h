/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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

#include "InlineStyleSheetOwner.h"
#include "SVGElement.h"
#include "Timer.h"

namespace WebCore {

class SVGStyleElement final : public SVGElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGStyleElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGStyleElement);
public:
    static Ref<SVGStyleElement> create(const QualifiedName&, Document&, bool createdByParser);
    virtual ~SVGStyleElement();

    CSSStyleSheet* sheet() const { return m_styleSheetOwner.sheet(); }

    bool disabled() const;
    void setDisabled(bool);
                          
    const AtomString& type() const;
    void setType(const AtomString&);

    const AtomString& media() const;
    void setMedia(const AtomString&);

private:
    SVGStyleElement(const QualifiedName&, Document&, bool createdByParser);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;
    void childrenChanged(const ChildChange&) final;

    bool rendererIsNeeded(const RenderStyle&) final { return false; }
    bool supportsFocus() const final { return false; }

    void finishParsingChildren() final;

    virtual bool isLoading() const { return m_styleSheetOwner.isLoading(); }
    bool sheetLoaded() final { return m_styleSheetOwner.sheetLoaded(*this); }
    void startLoadingDynamicSheet() final { m_styleSheetOwner.startLoadingDynamicSheet(*this); }
    Timer* loadEventTimer() final { return &m_loadEventTimer; }

    String title() const final;

    InlineStyleSheetOwner m_styleSheetOwner;
    Timer m_loadEventTimer;
};

} // namespace WebCore
