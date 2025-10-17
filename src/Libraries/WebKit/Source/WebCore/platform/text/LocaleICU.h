/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#include "PlatformLocale.h"
#include <unicode/udat.h>
#include <unicode/unum.h>
#include <wtf/Forward.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

// We should use this class only for LocalizedNumberICU.cpp, LocalizedDateICU.cpp,
// and LocalizedNumberICUTest.cpp.
class LocaleICU final : public Locale {
public:
    explicit LocaleICU(const char*);
    virtual ~LocaleICU();

    Locale::WritingDirection defaultWritingDirection() const override;

    String dateFormat() override;
    String monthFormat() override;
    String shortMonthFormat() override;
    String timeFormat() override;
    String shortTimeFormat() override;
    String dateTimeFormatWithSeconds() override;
    String dateTimeFormatWithoutSeconds() override;
    const Vector<String>& monthLabels() override;
    const Vector<String>& shortMonthLabels() override;
    const Vector<String>& standAloneMonthLabels() override;
    const Vector<String>& shortStandAloneMonthLabels() override;
    const Vector<String>& timeAMPMLabels() override;

private:
#if !UCONFIG_NO_FORMATTING
    String decimalSymbol(UNumberFormatSymbol);
    String decimalTextAttribute(UNumberFormatTextAttribute);
#endif
    void initializeLocaleData() override;

    bool initializeShortDateFormat();
    UDateFormat* openDateFormat(UDateFormatStyle timeStyle, UDateFormatStyle dateStyle) const;

    std::unique_ptr<Vector<String>> createLabelVector(const UDateFormat*, UDateFormatSymbolType, int32_t startIndex, int32_t size);
    void initializeDateTimeFormat();

    CString m_locale;

#if !UCONFIG_NO_FORMATTING
    UNumberFormat* m_numberFormat { nullptr };
    bool m_didCreateDecimalFormat { false };
#endif

    std::unique_ptr<Vector<String>> m_monthLabels;
    String m_dateFormat;
    String m_monthFormat;
    String m_shortMonthFormat;
    String m_timeFormatWithSeconds;
    String m_timeFormatWithoutSeconds;
    String m_dateTimeFormatWithSeconds;
    String m_dateTimeFormatWithoutSeconds;
    UDateFormat* m_shortDateFormat { nullptr };
    UDateFormat* m_mediumTimeFormat { nullptr };
    UDateFormat* m_shortTimeFormat { nullptr };
    Vector<String> m_shortMonthLabels;
    Vector<String> m_standAloneMonthLabels;
    Vector<String> m_shortStandAloneMonthLabels;
    Vector<String> m_timeAMPMLabels;
    bool m_didCreateShortDateFormat { false };
    bool m_didCreateTimeFormat { false };
};

} // namespace WebCore
