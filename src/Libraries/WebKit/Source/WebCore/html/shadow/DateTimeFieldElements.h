/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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

#include "DateTimeNumericFieldElement.h"
#include "DateTimeSymbolicFieldElement.h"

namespace WebCore {

class DateTimeDayFieldElement final : public DateTimeNumericFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeDayFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeDayFieldElement);

public:
    static Ref<DateTimeDayFieldElement> create(Document&, DateTimeFieldElementFieldOwner&);

private:
    DateTimeDayFieldElement(Document&, DateTimeFieldElementFieldOwner&);

    // DateTimeFieldElement functions:
    void setValueAsDate(const DateComponents&) final;
    void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue) final;
};

class DateTimeHourFieldElement final : public DateTimeNumericFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeHourFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeHourFieldElement);

public:
    static Ref<DateTimeHourFieldElement> create(Document&, DateTimeFieldElementFieldOwner&, int minimum, int maximum);

private:
    DateTimeHourFieldElement(Document&, DateTimeFieldElementFieldOwner&, int minimum, int maximum);

    // DateTimeFieldElement functions:
    void setValueAsDate(const DateComponents&) final;
    void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue) final;
};

class DateTimeMeridiemFieldElement final : public DateTimeSymbolicFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeMeridiemFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeMeridiemFieldElement);

public:
    static Ref<DateTimeMeridiemFieldElement> create(Document&, DateTimeFieldElementFieldOwner&, const Vector<String>&);

private:
    DateTimeMeridiemFieldElement(Document&, DateTimeFieldElementFieldOwner&, const Vector<String>&);

    void updateAriaValueAttributes();
    // DateTimeFieldElement functions:
    void setEmptyValue(EventBehavior = DispatchNoEvent) final;
    void setValueAsDate(const DateComponents&) final;
    void setValueAsInteger(int, EventBehavior = DispatchNoEvent) final;

    void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue) final;
};

class DateTimeMillisecondFieldElement final : public DateTimeNumericFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeMillisecondFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeMillisecondFieldElement);

public:
    static Ref<DateTimeMillisecondFieldElement> create(Document&, DateTimeFieldElementFieldOwner&);

private:
    DateTimeMillisecondFieldElement(Document&, DateTimeFieldElementFieldOwner&);

    // DateTimeFieldElement functions:
    void setValueAsDate(const DateComponents&) final;
    void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue) final;
};

class DateTimeMinuteFieldElement final : public DateTimeNumericFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeMinuteFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeMinuteFieldElement);

public:
    static Ref<DateTimeMinuteFieldElement> create(Document&, DateTimeFieldElementFieldOwner&);

private:
    DateTimeMinuteFieldElement(Document&, DateTimeFieldElementFieldOwner&);

    // DateTimeFieldElement functions:
    void setValueAsDate(const DateComponents&) final;
    void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue) final;
};

class DateTimeMonthFieldElement final : public DateTimeNumericFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeMonthFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeMonthFieldElement);

public:
    static Ref<DateTimeMonthFieldElement> create(Document&, DateTimeFieldElementFieldOwner&);

private:
    DateTimeMonthFieldElement(Document&, DateTimeFieldElementFieldOwner&);

    // DateTimeFieldElement functions:
    void setValueAsDate(const DateComponents&) final;
    void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue) final;
};

class DateTimeSecondFieldElement final : public DateTimeNumericFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeSecondFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeSecondFieldElement);

public:
    static Ref<DateTimeSecondFieldElement> create(Document&, DateTimeFieldElementFieldOwner&);

private:
    DateTimeSecondFieldElement(Document&, DateTimeFieldElementFieldOwner&);

    // DateTimeFieldElement functions:
    void setValueAsDate(const DateComponents&) final;
    void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue) final;
};

class DateTimeSymbolicMonthFieldElement final : public DateTimeSymbolicFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeSymbolicMonthFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeSymbolicMonthFieldElement);

public:
    static Ref<DateTimeSymbolicMonthFieldElement> create(Document&, DateTimeFieldElementFieldOwner&, const Vector<String>&);

private:
    DateTimeSymbolicMonthFieldElement(Document&, DateTimeFieldElementFieldOwner&, const Vector<String>&);

    // DateTimeFieldElement functions:
    void setValueAsDate(const DateComponents&) final;
    void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue) final;
};

class DateTimeYearFieldElement final : public DateTimeNumericFieldElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DateTimeYearFieldElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DateTimeYearFieldElement);

public:
    static Ref<DateTimeYearFieldElement> create(Document&, DateTimeFieldElementFieldOwner&);

private:
    DateTimeYearFieldElement(Document&, DateTimeFieldElementFieldOwner&);

    // DateTimeFieldElement functions:
    void setValueAsDate(const DateComponents&) final;
    void populateDateTimeFieldsState(DateTimeFieldsState&, DateTimePlaceholderIfNoValue) final;
};

} // namespace WebCore
