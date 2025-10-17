/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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

#include "DateTimeFieldElement.h"
#include "TypeAhead.h"

namespace WebCore {

class DateTimeSymbolicFieldElement : public DateTimeFieldElement, public TypeAheadDataSource {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeSymbolicFieldElement);
protected:
    DateTimeSymbolicFieldElement(Document&, DateTimeFieldElementFieldOwner&, const Vector<String>&, int);
    size_t symbolsSize() const { return m_symbols.size(); }
    bool hasValue() const final;
    void setEmptyValue(EventBehavior = DispatchNoEvent) override;
    void setValueAsInteger(int, EventBehavior = DispatchNoEvent) override;
    int valueAsInteger() const final { return m_selectedIndex; }
    int placeholderValueAsInteger() const final { return m_placeholderIndex; }

private:
    static constexpr int invalidIndex = -1;

    // DateTimeFieldElement functions:
    void adjustMinInlineSize(RenderStyle&) const final;
    void stepDown() final;
    void stepUp() final;
    String value() const final;
    String placeholderValue() const final;
    void handleKeyboardEvent(KeyboardEvent&) final;

    // TypeAheadDataSource functions:
    int indexOfSelectedOption() const final;
    int optionCount() const final;
    String optionAtIndex(int index) const final;

    const Vector<String> m_symbols;
    int m_selectedIndex { invalidIndex };
    int m_placeholderIndex { invalidIndex };

    TypeAhead m_typeAhead;
};

} // namespace WebCore
