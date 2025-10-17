/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#include "PseudoElement.h"

#include "ContentData.h"
#include "InspectorInstrumentation.h"
#include "KeyframeEffectStack.h"
#include "RenderElement.h"
#include "RenderImage.h"
#include "RenderQuote.h"
#include "RenderStyleInlines.h"
#include "StyleResolver.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PseudoElement);

const QualifiedName& pseudoElementTagName()
{
    static NeverDestroyed<QualifiedName> name(nullAtom(), "<pseudo>"_s, nullAtom());
    return name;
}

PseudoElement::PseudoElement(Element& host, PseudoId pseudoId)
    : Element(pseudoElementTagName(), host.document(), TypeFlag::HasCustomStyleResolveCallbacks)
    , m_hostElement(host)
    , m_pseudoId(pseudoId)
{
    setEventTargetFlag(EventTargetFlag::IsConnected);
    ASSERT(pseudoId == PseudoId::Before || pseudoId == PseudoId::After);
}

PseudoElement::~PseudoElement()
{
    ASSERT(!m_hostElement);
}

Ref<PseudoElement> PseudoElement::create(Element& host, PseudoId pseudoId)
{
    Ref pseudoElement = adoptRef(*new PseudoElement(host, pseudoId));

    InspectorInstrumentation::pseudoElementCreated(host.document().protectedPage().get(), pseudoElement.get());

    return pseudoElement;
}

void PseudoElement::clearHostElement()
{
    InspectorInstrumentation::pseudoElementDestroyed(document().protectedPage().get(), *this);

    Styleable::fromElement(*this).elementWasRemoved();

    m_hostElement = nullptr;
}

bool PseudoElement::rendererIsNeeded(const RenderStyle& style)
{
    if (pseudoElementRendererIsNeeded(&style))
        return true;

    if (RefPtr element = m_hostElement.get()) {
        if (auto* stack = element->keyframeEffectStack(Style::PseudoElementIdentifier { pseudoId() }))
            return stack->requiresPseudoElement();
    }
    return false;
}

} // namespace
