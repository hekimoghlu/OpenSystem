/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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

#if ENABLE(MATHML)

#include "MathMLRowElement.h"

namespace WebCore {

class MathMLScriptsElement : public MathMLRowElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MathMLScriptsElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MathMLScriptsElement);
public:
    static Ref<MathMLScriptsElement> create(const QualifiedName& tagName, Document&);

    enum class ScriptType { Sub, Super, SubSup, Multiscripts, Under, Over, UnderOver };
    ScriptType scriptType() const { return m_scriptType; }
    const Length& subscriptShift();
    const Length& superscriptShift();

protected:
    MathMLScriptsElement(const QualifiedName& tagName, Document&);

private:
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    bool acceptsMathVariantAttribute() override { return false; };
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;

    const ScriptType m_scriptType;
    std::optional<Length> m_subscriptShift;
    std::optional<Length> m_superscriptShift;
};

}

#endif // ENABLE(MATHML)
