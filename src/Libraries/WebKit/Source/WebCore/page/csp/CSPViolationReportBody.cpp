/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 8, 2021.
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
#include "CSPViolationReportBody.h"

#include "ContentSecurityPolicyClient.h"
#include "FormData.h"
#include "SecurityPolicyViolationEvent.h"
#include <wtf/JSONValues.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using Init = SecurityPolicyViolationEventInit;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSPViolationReportBody);

CSPViolationReportBody::CSPViolationReportBody(Init&& init)
    : m_documentURL(WTFMove(init.documentURI))
    , m_referrer(init.referrer.isNull() ? emptyString() : WTFMove(init.referrer))
    , m_blockedURL(WTFMove(init.blockedURI))
    , m_effectiveDirective(WTFMove(init.effectiveDirective))
    , m_originalPolicy(WTFMove(init.originalPolicy))
    , m_sourceFile(WTFMove(init.sourceFile))
    , m_sample(WTFMove(init.sample))
    , m_disposition(init.disposition)
    , m_statusCode(init.statusCode)
    , m_lineNumber(init.lineNumber)
    , m_columnNumber(init.columnNumber)
{
}

CSPViolationReportBody::CSPViolationReportBody(String&& documentURL, String&& referrer, String&& blockedURL, String&& effectiveDirective, String&& originalPolicy, String&& sourceFile, String&& sample, SecurityPolicyViolationEventDisposition disposition, unsigned short statusCode, unsigned long lineNumber, unsigned long columnNumber)
    : m_documentURL(WTFMove(documentURL))
    , m_referrer(WTFMove(referrer))
    , m_blockedURL(WTFMove(blockedURL))
    , m_effectiveDirective(WTFMove(effectiveDirective))
    , m_originalPolicy(WTFMove(originalPolicy))
    , m_sourceFile(WTFMove(sourceFile))
    , m_sample(WTFMove(sample))
    , m_disposition(disposition)
    , m_statusCode(statusCode)
    , m_lineNumber(lineNumber)
    , m_columnNumber(columnNumber)
{
}

Ref<CSPViolationReportBody> CSPViolationReportBody::create(Init&& init)
{
    return adoptRef(*new CSPViolationReportBody(WTFMove(init)));
}

Ref<CSPViolationReportBody> CSPViolationReportBody::create(String&& documentURL, String&& referrer, String&& blockedURL, String&& effectiveDirective, String&& originalPolicy, String&& sourceFile, String&& sample, SecurityPolicyViolationEventDisposition disposition, unsigned short statusCode, unsigned long lineNumber, unsigned long columnNumber)
{
    return adoptRef(*new CSPViolationReportBody(WTFMove(documentURL), WTFMove(referrer), WTFMove(blockedURL), WTFMove(effectiveDirective), WTFMove(originalPolicy), WTFMove(sourceFile), WTFMove(sample), disposition, statusCode, lineNumber, columnNumber));
}

const String& CSPViolationReportBody::type() const
{
    static NeverDestroyed<const String> cspReportType(MAKE_STATIC_STRING_IMPL("csp-violation"));
    return cspReportType;
}

Ref<FormData> CSPViolationReportBody::createReportFormDataForViolation(bool usesReportTo, bool isReportOnly) const
{
    // We need to be careful here when deciding what information to send to the
    // report-uri. Currently, we send only the current document's URL and the
    // directive that was violated. The document's URL is safe to send because
    // it's the document itself that's requesting that it be sent. You could
    // make an argument that we shouldn't send HTTPS document URLs to HTTP
    // report-uris (for the same reasons that we suppress the Referer in that
    // case), but the Referrer is sent implicitly whereas this request is only
    // sent explicitly. As for which directive was violated, that's pretty
    // harmless information.

    auto cspReport = JSON::Object::create();

    if (usesReportTo) {
        // It looks like WPT expect the body for modern reports to use the same
        // syntax as the JSON object (not the hyphenated versions in the original
        // CSP spec.
        cspReport->setString("documentURL"_s, documentURL());
        cspReport->setString("disposition"_s, isReportOnly ? "report"_s : "enforce"_s);
        cspReport->setString("referrer"_s, referrer());
        cspReport->setString("effectiveDirective"_s, effectiveDirective());
        cspReport->setString("blockedURL"_s, blockedURL());
        cspReport->setString("originalPolicy"_s, originalPolicy());
        cspReport->setInteger("statusCode"_s, statusCode());
        cspReport->setString("sample"_s, sample());
        if (!sourceFile().isNull()) {
            cspReport->setString("sourceFile"_s, sourceFile());
            cspReport->setInteger("lineNumber"_s, lineNumber());
            cspReport->setInteger("columnNumber"_s, columnNumber());
        }
    } else {
        cspReport->setString("document-uri"_s, documentURL());
        cspReport->setString("referrer"_s, referrer());
        cspReport->setString("violated-directive"_s, effectiveDirective());
        cspReport->setString("effective-directive"_s, effectiveDirective());
        cspReport->setString("original-policy"_s, originalPolicy());
        cspReport->setString("blocked-uri"_s, blockedURL());
        cspReport->setInteger("status-code"_s, statusCode());
        if (!sourceFile().isNull()) {
            cspReport->setString("source-file"_s, sourceFile());
            cspReport->setInteger("line-number"_s, lineNumber());
            cspReport->setInteger("column-number"_s, columnNumber());
        }
    }

    // https://www.w3.org/TR/reporting-1/#queue-report, step 2.3.1.
    auto reportObject = JSON::Object::create();
    reportObject->setString("type"_s, type());
    reportObject->setString("url"_s, documentURL());
    reportObject->setObject(usesReportTo ? "body"_s : "csp-report"_s, WTFMove(cspReport));

    return FormData::create(reportObject->toJSONString().utf8());
}

} // namespace WebCore
