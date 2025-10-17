/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#include "DeprecationReportBody.h"

#include "DateComponents.h"
#include "FormData.h"
#include <wtf/JSONValues.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DeprecationReportBody);

DeprecationReportBody::DeprecationReportBody(String&& id, WallTime anticipatedRemoval, String&& message, String&& sourceFile, std::optional<unsigned> lineNumber, std::optional<unsigned> columnNumber)
    : m_id(WTFMove(id))
    , m_anticipatedRemoval(anticipatedRemoval)
    , m_message(WTFMove(message))
    , m_sourceFile(WTFMove(sourceFile))
    , m_lineNumber(lineNumber)
    , m_columnNumber(columnNumber)
{
}

Ref<DeprecationReportBody> DeprecationReportBody::create(String&& id, WallTime anticipatedRemoval, String&& message, String&& sourceFile, std::optional<unsigned> lineNumber, std::optional<unsigned> columnNumber)
{
    return adoptRef(*new DeprecationReportBody(WTFMove(id), anticipatedRemoval, WTFMove(message), WTFMove(sourceFile), lineNumber, columnNumber));
}

const String& DeprecationReportBody::type() const
{
    static NeverDestroyed<const String> reportType(MAKE_STATIC_STRING_IMPL("deprecation"));
    return reportType;
}

Ref<FormData> DeprecationReportBody::createReportFormDataForViolation() const
{
    // https://wicg.github.io/deprecation-reporting/#deprecation-report
    // Suitable for network endpoints.
    auto reportBody = JSON::Object::create();
    reportBody->setString("id"_s, m_id);
    reportBody->setString("anticipatedRemoval"_s, DateComponents::fromMillisecondsSinceEpochForDate(m_anticipatedRemoval.secondsSinceEpoch().milliseconds())->toString());
    reportBody->setString("message"_s, m_message);
    if (!m_sourceFile.isNull()) {
        reportBody->setString("sourceFile"_s, m_sourceFile);
        reportBody->setInteger("lineNumber"_s, m_lineNumber.value_or(0));
        reportBody->setInteger("columnNumber"_s, m_columnNumber.value_or(0));
    }

    auto reportObject = JSON::Object::create();
    reportObject->setString("type"_s, type());
    reportObject->setString("url"_s, ""_s);
    reportObject->setObject("body"_s, WTFMove(reportBody));

    return FormData::create(reportObject->toJSONString().utf8());
}

} // namespace WebCore
