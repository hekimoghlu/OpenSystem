/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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

#include "AccessibilityRenderObject.h"
#include "RenderMathMLBlock.h"
#include "RenderMathMLFraction.h"
#include "RenderMathMLMath.h"
#include "RenderMathMLOperator.h"
#include "RenderMathMLRoot.h"

namespace WebCore {

class AccessibilityMathMLElement : public AccessibilityRenderObject {

public:
    static Ref<AccessibilityMathMLElement> create(AXID, RenderObject&, bool isAnonymousOperator);
    virtual ~AccessibilityMathMLElement();

protected:
    explicit AccessibilityMathMLElement(AXID, RenderObject&, bool isAnonymousOperator);

private:
    AccessibilityRole determineAccessibilityRole() final;
    void addChildren() final;
    String textUnderElement(TextUnderElementMode = TextUnderElementMode()) const final;
    String stringValue() const final;
    bool isIgnoredElementWithinMathTree() const final;

    bool isMathElement() const final { return true; }

    bool isMathFraction() const final;
    bool isMathFenced() const final;
    bool isMathSubscriptSuperscript() const final;
    bool isMathRow() const final;
    bool isMathUnderOver() const final;
    bool isMathRoot() const final;
    bool isMathSquareRoot() const final;
    bool isMathText() const final;
    bool isMathNumber() const final;
    bool isMathOperator() const final;
    bool isMathFenceOperator() const final;
    bool isMathSeparatorOperator() const final;
    bool isMathIdentifier() const final;
    bool isMathTable() const final;
    bool isMathTableRow() const final;
    bool isMathTableCell() const final;
    bool isMathMultiscript() const final;
    bool isMathToken() const final;
    bool isMathScriptObject(AccessibilityMathScriptObjectType) const final;
    bool isMathMultiscriptObject(AccessibilityMathMultiscriptObjectType) const final;

    // Generic components.
    AXCoreObject* mathBaseObject() final;

    // Root components.
    std::optional<AccessibilityChildrenVector> mathRadicand() final;
    AXCoreObject* mathRootIndexObject() final;

    // Fraction components.
    AXCoreObject* mathNumeratorObject() final;
    AXCoreObject* mathDenominatorObject() final;

    // Under over components.
    AXCoreObject* mathUnderObject() final;
    AXCoreObject* mathOverObject() final;

    // Subscript/superscript components.
    AXCoreObject* mathSubscriptObject() final;
    AXCoreObject* mathSuperscriptObject() final;

    // Fenced components.
    String mathFencedOpenString() const final;
    String mathFencedCloseString() const final;
    int mathLineThickness() const final;
    bool isAnonymousMathOperator() const final;

    // Multiscripts components.
    void mathPrescripts(AccessibilityMathMultiscriptPairs&) final;
    void mathPostscripts(AccessibilityMathMultiscriptPairs&) final;

    bool m_isAnonymousOperator;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ACCESSIBILITY(AccessibilityMathMLElement, isMathElement())

#endif // ENABLE(MATHML)
