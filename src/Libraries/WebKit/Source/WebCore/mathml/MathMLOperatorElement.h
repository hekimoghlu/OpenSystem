/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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

#include "MathMLOperatorDictionary.h"
#include "MathMLTokenElement.h"

namespace WebCore {

class MathMLOperatorElement final : public MathMLTokenElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MathMLOperatorElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MathMLOperatorElement);
public:
    static Ref<MathMLOperatorElement> create(const QualifiedName& tagName, Document&);
    struct OperatorChar {
        char32_t character { 0 };
        bool isVertical { true };
    };
    static OperatorChar parseOperatorChar(const String&);
    const OperatorChar& operatorChar();
    void setOperatorFormDirty() { m_dictionaryProperty = std::nullopt; }
    MathMLOperatorDictionary::Form form() { return dictionaryProperty().form; }
    bool hasProperty(MathMLOperatorDictionary::Flag);
    Length defaultLeadingSpace();
    Length defaultTrailingSpace();
    const Length& leadingSpace();
    const Length& trailingSpace();
    const Length& minSize();
    const Length& maxSize();

private:
    MathMLOperatorElement(const QualifiedName& tagName, Document&);
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    void childrenChanged(const ChildChange&) final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;

    std::optional<OperatorChar> m_operatorChar;

    std::optional<MathMLOperatorDictionary::Property> m_dictionaryProperty;
    MathMLOperatorDictionary::Property computeDictionaryProperty();
    const MathMLOperatorDictionary::Property& dictionaryProperty();

    struct OperatorProperties {
        unsigned short flags;
        unsigned short dirtyFlags { MathMLOperatorDictionary::allFlags };
    };
    OperatorProperties m_properties;
    void computeOperatorFlag(MathMLOperatorDictionary::Flag);

    std::optional<Length> m_leadingSpace;
    std::optional<Length> m_trailingSpace;
    std::optional<Length> m_minSize;
    std::optional<Length> m_maxSize;
};

}

#endif // ENABLE(MATHML)
