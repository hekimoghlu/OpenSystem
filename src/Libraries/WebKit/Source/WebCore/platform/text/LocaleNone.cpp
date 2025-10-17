/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 18, 2023.
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
#include "config.h"
#include "PlatformLocale.h"
#include <wtf/DateMath.h>

namespace WebCore {

class LocaleNone : public Locale {
public:
    virtual ~LocaleNone();

private:
    void initializeLocaleData() final;
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

    Vector<String> m_timeAMPMLabels;
    Vector<String> m_shortMonthLabels;
    Vector<String> m_monthLabels;
};

std::unique_ptr<Locale> Locale::create(const AtomString&)
{
    return makeUnique<LocaleNone>();
}

LocaleNone::~LocaleNone() = default;

void LocaleNone::initializeLocaleData()
{
}

const Vector<String>& LocaleNone::monthLabels()
{
    if (!m_monthLabels.isEmpty())
        return m_monthLabels;
    m_monthLabels = { WTF::monthFullName, std::size(WTF::monthFullName) };
    return m_monthLabels;
}

String LocaleNone::dateFormat()
{
    return "yyyy-MM-dd"_s;
}

String LocaleNone::monthFormat()
{
    return "yyyy-MM"_s;
}

String LocaleNone::shortMonthFormat()
{
    return "yyyy-MM"_s;
}

String LocaleNone::timeFormat()
{
    return "HH:mm:ss"_s;
}

String LocaleNone::shortTimeFormat()
{
    return "HH:mm"_s;
}

String LocaleNone::dateTimeFormatWithSeconds()
{
    return "yyyy-MM-dd'T'HH:mm:ss"_s;
}

String LocaleNone::dateTimeFormatWithoutSeconds()
{
    return "yyyy-MM-dd'T'HH:mm"_s;
}

const Vector<String>& LocaleNone::shortMonthLabels()
{
    if (!m_shortMonthLabels.isEmpty())
        return m_shortMonthLabels;
    m_shortMonthLabels = { WTF::monthName, std::size(WTF::monthName) };
    return m_shortMonthLabels;
}

const Vector<String>& LocaleNone::shortStandAloneMonthLabels()
{
    return shortMonthLabels();
}

const Vector<String>& LocaleNone::standAloneMonthLabels()
{
    return monthLabels();
}

const Vector<String>& LocaleNone::timeAMPMLabels()
{
    if (!m_timeAMPMLabels.isEmpty())
        return m_timeAMPMLabels;
    m_timeAMPMLabels.appendList({ "AM"_s, "PM"_s });
    return m_timeAMPMLabels;
}

} // namespace WebCore
