/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
#include "Report.h"

#include "FormData.h"
#include "ReportBody.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Report);

Ref<Report> Report::create(const String& type, const String& url, RefPtr<ReportBody>&& body)
{
    return adoptRef(*new Report(type, url, WTFMove(body)));
}

Report::Report(const String& type, const String& url, RefPtr<ReportBody>&& body)
    : m_type(type)
    , m_url(url)
    , m_body(WTFMove(body))
{
}

Report::~Report() = default;

const String& Report::type() const
{
    return m_type;
}

const String& Report::url() const
{
    return m_url;
}

const RefPtr<ReportBody>& Report::body() const
{
    return m_body;
}

Ref<FormData> Report::createReportFormDataForViolation(const String& type, const URL& url, const String& userAgent, const String& destination, const Function<void(JSON::Object&)>& populateBody)
{
    auto body = JSON::Object::create();
    populateBody(body);

    // https://www.w3.org/TR/reporting-1/#queue-report, step 2.3.1.
    auto reportObject = JSON::Object::create();
    reportObject->setObject("body"_s, WTFMove(body));
    reportObject->setString("user_agent"_s, userAgent);
    reportObject->setString("destination"_s, destination);
    reportObject->setString("type"_s, type);
    reportObject->setInteger("age"_s, 0); // We currently do not delay sending the reports.
    reportObject->setInteger("attempts"_s, 0);
    if (url.isValid())
        reportObject->setString("url"_s, url.string());

    auto reportList = JSON::Array::create();
    reportList->pushObject(reportObject);

    return FormData::create(reportList->toJSONString().utf8());
}

} // namespace WebCore
