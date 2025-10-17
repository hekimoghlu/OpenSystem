/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(MAC)
#define PlatformNSParagraphStyle NSParagraphStyle.class
#else
#define PlatformNSParagraphStyle PAL::getNSParagraphStyleClass()
#endif

OBJC_CLASS NSCalendar;
OBJC_CLASS NSDateFormatter;
OBJC_CLASS NSParagraphStyle;
OBJC_CLASS NSLocale;

namespace WebCore {

class DateComponents;

class LocaleCocoa final : public Locale {
    WTF_MAKE_TZONE_ALLOCATED(LocaleCocoa);
public:
    explicit LocaleCocoa(const AtomString&);
    ~LocaleCocoa();

    Locale::WritingDirection defaultWritingDirection() const override;

    String formatDateTime(const DateComponents&, FormatType = FormatTypeUnspecified) override;

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

    static RetainPtr<CFStringRef> canonicalLanguageIdentifierFromString(const AtomString&);
    static void releaseMemory();

private:
    RetainPtr<NSDateFormatter> shortDateFormatter();
    void initializeLocaleData() override;

    RetainPtr<NSLocale> m_locale;
    RetainPtr<NSCalendar> m_gregorianCalendar;
    Vector<String> m_monthLabels;
    RetainPtr<NSDateFormatter> timeFormatter();
    RetainPtr<NSDateFormatter> shortTimeFormatter();
    RetainPtr<NSDateFormatter> dateTimeFormatterWithSeconds();
    RetainPtr<NSDateFormatter> dateTimeFormatterWithoutSeconds();

    String m_dateFormat;
    String m_monthFormat;
    String m_shortMonthFormat;
    String m_timeFormatWithSeconds;
    String m_timeFormatWithoutSeconds;
    String m_dateTimeFormatWithSeconds;
    String m_dateTimeFormatWithoutSeconds;
    Vector<String> m_shortMonthLabels;
    Vector<String> m_standAloneMonthLabels;
    Vector<String> m_shortStandAloneMonthLabels;
    Vector<String> m_timeAMPMLabels;
    bool m_didInitializeNumberData;
};

} // namespace WebCore
