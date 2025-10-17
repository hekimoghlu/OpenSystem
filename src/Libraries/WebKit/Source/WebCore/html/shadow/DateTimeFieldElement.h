/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

#include "HTMLDivElement.h"

#include <wtf/GregorianDateTime.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class DateTimeFieldElementFieldOwner;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::DateTimeFieldElementFieldOwner> : std::true_type { };
}

namespace WebCore {

class DateComponents;
class DateTimeFieldElement;
class RenderStyle;

struct DateTimeFieldsState;

enum class DateTimePlaceholderIfNoValue : bool { No, Yes };

class DateTimeFieldElementFieldOwner : public CanMakeWeakPtr<DateTimeFieldElementFieldOwner> {
public:
    virtual ~DateTimeFieldElementFieldOwner();
    virtual void didBlurFromField(Event&) = 0;
    virtual void fieldValueChanged() = 0;
    virtual bool focusOnNextField(const DateTimeFieldElement&) = 0;
    virtual bool focusOnPreviousField(const DateTimeFieldElement&) = 0;
    virtual bool isFieldOwnerDisabled() const = 0;
    virtual bool isFieldOwnerReadOnly() const = 0;
    virtual bool isFieldOwnerHorizontal() const = 0;
    virtual AtomString localeIdentifier() const = 0;
    virtual const GregorianDateTime& placeholderDate() const = 0;
};

class DateTimeFieldElement : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeFieldElement);
public:
    enum EventBehavior : bool { DispatchNoEvent, DispatchInputAndChangeEvents };

    void defaultEventHandler(Event&) override;
    bool isFocusable() const final;

    String visibleValue() const;

    virtual bool hasValue() const = 0;
    virtual void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue = DateTimePlaceholderIfNoValue::No) = 0;
    virtual void setEmptyValue(EventBehavior = DispatchNoEvent) = 0;
    virtual void setValueAsDate(const DateComponents&) = 0;
    virtual void setValueAsInteger(int, EventBehavior = DispatchNoEvent) = 0;
    virtual void stepDown() = 0;
    virtual void stepUp() = 0;
    virtual String value() const = 0;
    virtual String placeholderValue() const = 0;

protected:
    DateTimeFieldElement(Document&, DateTimeFieldElementFieldOwner&);
    Locale& localeForOwner() const;
    AtomString localeIdentifier() const;
    void updateVisibleValue(EventBehavior);
    virtual void adjustMinInlineSize(RenderStyle&) const = 0;
    virtual int valueAsInteger() const = 0;
    virtual int placeholderValueAsInteger() const = 0;
    virtual void handleKeyboardEvent(KeyboardEvent&) = 0;
    virtual void handleBlurEvent(Event&);

private:
    std::optional<Style::ResolvedStyle> resolveCustomStyle(const Style::ResolutionContext&, const RenderStyle*) final;

    bool supportsFocus() const override;

    void defaultKeyboardEventHandler(KeyboardEvent&);
    bool isFieldOwnerDisabled() const;
    bool isFieldOwnerReadOnly() const;
    bool isFieldOwnerHorizontal() const;

    WeakPtr<DateTimeFieldElementFieldOwner> m_fieldOwner;
};

} // namespace WebCore
