/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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

#include "DateTimeChooser.h"
#include "DateTimeChooserClient.h"
#include "DateTimeEditElement.h"
#include "DateTimeFormat.h"
#include "InputType.h"
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DateComponents;

struct DateTimeChooserParameters;

// A super class of date, datetime, datetime-local, month, time, and week types.
class BaseDateAndTimeInputType : public InputType, public DateTimeChooserClient, private DateTimeEditElementEditControlOwner {
    WTF_MAKE_TZONE_ALLOCATED(BaseDateAndTimeInputType);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(BaseDateAndTimeInputType);
public:
    bool typeMismatchFor(const String&) const final;
    bool valueMissing(const String&) const final;
    bool typeMismatch() const final;
    bool hasBadInput() const final;

protected:
    enum class DateTimeFormatValidationResults : uint8_t {
        HasYear = 1 << 0,
        HasMonth = 1 << 1,
        HasWeek = 1 << 2,
        HasDay = 1 << 3,
        HasHour = 1 << 4,
        HasMinute = 1 << 5,
        HasSecond = 1 << 6,
        HasMeridiem = 1 << 7,
    };

    BaseDateAndTimeInputType(Type type, HTMLInputElement& element)
        : InputType(type, element)
    {
        ASSERT(needsShadowSubtree());
    }

    ~BaseDateAndTimeInputType();

    Decimal parseToNumber(const String&, const Decimal&) const override;
    String serialize(const Decimal&) const final;
    String serializeWithComponents(const DateComponents&) const;

    bool shouldHaveSecondField(const DateComponents&) const;
    bool shouldHaveMillisecondField(const DateComponents&) const;

private:
    class DateTimeFormatValidator final : public DateTimeFormat::TokenHandler {
    public:
        DateTimeFormatValidator() { }

        void visitField(DateTimeFormat::FieldType, int);
        void visitLiteral(String&&) { }

        bool validateFormat(const String& format, const BaseDateAndTimeInputType&);

    private:
        OptionSet<DateTimeFormatValidationResults> m_results;
    };

    virtual std::optional<DateComponents> parseToDateComponents(StringView) const = 0;
    virtual std::optional<DateComponents> setMillisecondToDateComponents(double) const = 0;
    virtual void setupLayoutParameters(DateTimeEditElement::LayoutParameters&, const DateComponents&) const = 0;
    virtual bool isValidFormat(OptionSet<DateTimeFormatValidationResults>) const = 0;
    virtual String serializeWithMilliseconds(double) const;

    // InputType functions:
    String visibleValue() const final;
    String sanitizeValue(const String&) const override;
    void setValue(const String&, bool valueChanged, TextFieldEventBehavior, TextControlSetValueSelection) final;
    WallTime valueAsDate() const override;
    ExceptionOr<void> setValueAsDate(WallTime) const override;
    WallTime accessibilityValueAsDate() const final;
    double valueAsDouble() const final;
    ExceptionOr<void> setValueAsDecimal(const Decimal&, TextFieldEventBehavior) const final;
    Decimal defaultValueForStepUp() const override;
    String localizeValue(const String&) const final;
    bool supportsReadOnly() const final;
    bool shouldRespectListAttribute() final;
    bool isKeyboardFocusable(KeyboardEvent*) const final;
    bool isMouseFocusable() const final;

    void handleDOMActivateEvent(Event&) override;
    void createShadowSubtree() final;
    void removeShadowSubtree() final;
    void updateInnerTextValue() final;
    bool hasCustomFocusLogic() const final;
    void attributeChanged(const QualifiedName&) final;
    bool isPresentingAttachedView() const final;
    void elementDidBlur() final;
    void detach() final;

    ShouldCallBaseEventHandler handleKeydownEvent(KeyboardEvent&) final;
    void handleKeypressEvent(KeyboardEvent&) final;
    void handleKeyupEvent(KeyboardEvent&) final;
    void handleFocusEvent(Node* oldFocusedNode, FocusDirection) final;
    bool accessKeyAction(bool sendMouseEvents) final;

    // DateTimeEditElementEditControlOwner functions:
    void didBlurFromControl() final;
    void didChangeValueFromControl() final;
    bool isEditControlOwnerDisabled() const final;
    bool isEditControlOwnerReadOnly() const final;
    AtomString localeIdentifier() const final;

    // DateTimeChooserClient functions:
    void didChooseValue(StringView) final;
    void didEndChooser() final { m_dateTimeChooser = nullptr; }

    bool setupDateTimeChooserParameters(DateTimeChooserParameters&);
    void closeDateTimeChooser();

    void showPicker() override;

    RefPtr<DateTimeChooser> m_dateTimeChooser;
    RefPtr<DateTimeEditElement> m_dateTimeEditElement;
};

} // namespace WebCore
