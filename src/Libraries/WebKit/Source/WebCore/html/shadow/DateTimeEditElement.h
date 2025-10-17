/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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

#include "DateComponents.h"
#include "DateTimeFieldElement.h"

#include <wtf/WeakPtr.h>

namespace WebCore {
class DateTimeEditElementEditControlOwner;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::DateTimeEditElementEditControlOwner> : std::true_type { };
}

namespace WebCore {

class Locale;

class DateTimeEditElementEditControlOwner : public CanMakeWeakPtr<DateTimeEditElementEditControlOwner> {
public:
    virtual ~DateTimeEditElementEditControlOwner();
    virtual void didBlurFromControl() = 0;
    virtual void didChangeValueFromControl() = 0;
    virtual String formatDateTimeFieldsState(const DateTimeFieldsState&) const = 0;
    virtual bool isEditControlOwnerDisabled() const = 0;
    virtual bool isEditControlOwnerReadOnly() const = 0;
    virtual AtomString localeIdentifier() const = 0;
};

class DateTimeEditElement final : public HTMLDivElement, public DateTimeFieldElementFieldOwner {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeEditElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeEditElement);
public:
    struct LayoutParameters {
        String dateTimeFormat;
        String fallbackDateTimeFormat;
        Locale& locale;
        bool shouldHaveMillisecondField { false };

        LayoutParameters(Locale& locale)
            : locale(locale)
        {
        }
    };

    static Ref<DateTimeEditElement> create(Document&, DateTimeEditElementEditControlOwner&);

    virtual ~DateTimeEditElement();
    void addField(Ref<DateTimeFieldElement>);
    Element& fieldsWrapperElement() const;
    void focusByOwner();
    void resetFields();
    void setEmptyValue(const LayoutParameters&);
    void setValueAsDate(const LayoutParameters&, const DateComponents&);
    String value() const;
    String placeholderValue() const;
    bool editableFieldsHaveValues() const;

private:
    // Datetime can be represented by at most 8 fields:
    // 1. year
    // 2. month
    // 3. day-of-month
    // 4. hour
    // 5. minute
    // 6. second
    // 7. millisecond
    // 8. AM/PM
    static constexpr int maximumNumberOfFields = 8;

    DateTimeEditElement(Document&, DateTimeEditElementEditControlOwner&);

    size_t fieldIndexOf(const DateTimeFieldElement&) const;
    DateTimeFieldElement* focusedFieldElement() const;
    void layout(const LayoutParameters&);
    DateTimeFieldsState valueAsDateTimeFieldsState(DateTimePlaceholderIfNoValue = DateTimePlaceholderIfNoValue::No) const;

    bool focusOnNextFocusableField(size_t startIndex);

    // DateTimeFieldElementFieldOwner functions:
    void didBlurFromField(Event&) final;
    void fieldValueChanged() final;
    bool focusOnNextField(const DateTimeFieldElement&) final;
    bool focusOnPreviousField(const DateTimeFieldElement&) final;
    bool isFieldOwnerDisabled() const final;
    bool isFieldOwnerReadOnly() const final;
    bool isFieldOwnerHorizontal() const final;
    AtomString localeIdentifier() const final;
    const GregorianDateTime& placeholderDate() const final;

    Vector<Ref<DateTimeFieldElement>, maximumNumberOfFields> m_fields;
    WeakPtr<DateTimeEditElementEditControlOwner> m_editControlOwner;
    GregorianDateTime m_placeholderDate;
};

} // namespace WebCore
