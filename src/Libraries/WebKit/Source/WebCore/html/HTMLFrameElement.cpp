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
#include "config.h"
#include "HTMLFrameElement.h"

#include "ElementInlines.h"
#include "HTMLFrameSetElement.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "LocalFrame.h"
#include "RenderFrame.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLFrameElement);

using namespace HTMLNames;

inline HTMLFrameElement::HTMLFrameElement(const QualifiedName& tagName, Document& document)
    : HTMLFrameElementBase(tagName, document)
{
    ASSERT(hasTagName(frameTag));
}

Ref<HTMLFrameElement> HTMLFrameElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLFrameElement(tagName, document));
}

bool HTMLFrameElement::rendererIsNeeded(const RenderStyle& style)
{
    return HTMLFrameElementBase::rendererIsNeeded(style) && canLoad();
}

RenderPtr<RenderElement> HTMLFrameElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    return createRenderer<RenderFrame>(*this, WTFMove(style));
}

bool HTMLFrameElement::noResize() const
{
    return hasAttributeWithoutSynchronization(noresizeAttr);
}

void HTMLFrameElement::didAttachRenderers()
{
    HTMLFrameElementBase::didAttachRenderers();
    const auto containingFrameSet = HTMLFrameSetElement::findContaining(this);
    if (!containingFrameSet)
        return;
    if (!m_frameBorderSet)
        m_frameBorder = containingFrameSet->hasFrameBorder();
}

int HTMLFrameElement::defaultTabIndex() const
{
    return 0;
}

void HTMLFrameElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == frameborderAttr) {
        m_frameBorder = parseHTMLInteger(newValue).value_or(0);
        m_frameBorderSet = !newValue.isNull();
        // FIXME: If we are already attached, this has no effect.
    } else if (name == noresizeAttr) {
        if (auto* renderer = this->renderer())
            renderer->updateFromElement();
    } else
        HTMLFrameElementBase::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

} // namespace WebCore
