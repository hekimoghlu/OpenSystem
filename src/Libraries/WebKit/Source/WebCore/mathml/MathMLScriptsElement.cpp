/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#include "MathMLScriptsElement.h"

#if ENABLE(MATHML)

#include "RenderMathMLScripts.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MathMLScriptsElement);

using namespace MathMLNames;

static MathMLScriptsElement::ScriptType scriptTypeOf(const QualifiedName& tagName)
{
    if (tagName.matches(msubTag))
        return MathMLScriptsElement::ScriptType::Sub;
    if (tagName.matches(msupTag))
        return MathMLScriptsElement::ScriptType::Super;
    if (tagName.matches(msubsupTag))
        return MathMLScriptsElement::ScriptType::SubSup;
    if (tagName.matches(munderTag))
        return MathMLScriptsElement::ScriptType::Under;
    if (tagName.matches(moverTag))
        return MathMLScriptsElement::ScriptType::Over;
    if (tagName.matches(munderoverTag))
        return MathMLScriptsElement::ScriptType::UnderOver;
    ASSERT(tagName.matches(mmultiscriptsTag));
    return MathMLScriptsElement::ScriptType::Multiscripts;
}

MathMLScriptsElement::MathMLScriptsElement(const QualifiedName& tagName, Document& document)
    : MathMLRowElement(tagName, document)
    , m_scriptType(scriptTypeOf(tagName))
{
}

Ref<MathMLScriptsElement> MathMLScriptsElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new MathMLScriptsElement(tagName, document));
}

const MathMLElement::Length& MathMLScriptsElement::subscriptShift()
{
    return cachedMathMLLength(subscriptshiftAttr, m_subscriptShift);
}

const MathMLElement::Length& MathMLScriptsElement::superscriptShift()
{
    return cachedMathMLLength(superscriptshiftAttr, m_superscriptShift);
}

void MathMLScriptsElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == subscriptshiftAttr)
        m_subscriptShift = std::nullopt;
    else if (name == superscriptshiftAttr)
        m_superscriptShift = std::nullopt;

    MathMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

RenderPtr<RenderElement> MathMLScriptsElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    ASSERT(hasTagName(msubTag) || hasTagName(msupTag) || hasTagName(msubsupTag) || hasTagName(mmultiscriptsTag));
    return createRenderer<RenderMathMLScripts>(RenderObject::Type::MathMLScripts, *this, WTFMove(style));
}

}

#endif // ENABLE(MATHML)
