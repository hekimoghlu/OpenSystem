/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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

#include "Element.h"
#include <wtf/Forward.h>

namespace WebCore {

class PseudoElement final : public Element {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PseudoElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PseudoElement);
public:
    static Ref<PseudoElement> create(Element& host, PseudoId);
    virtual ~PseudoElement();

    Element* hostElement() const { return m_hostElement.get(); }
    void clearHostElement();

    bool rendererIsNeeded(const RenderStyle&) override;

    bool canStartSelection() const override { return false; }
    bool canContainRangeEndPoint() const override { return false; }

private:
    PseudoElement(Element&, PseudoId);

    PseudoId customPseudoId() const override { return m_pseudoId; }

    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_hostElement;
    PseudoId m_pseudoId;
};

const QualifiedName& pseudoElementTagName();

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::PseudoElement)
    static bool isType(const WebCore::Node& node) { return node.isPseudoElement(); }
SPECIALIZE_TYPE_TRAITS_END()
