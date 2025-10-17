/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#include "CrossOriginEmbedderPolicy.h"

#include "COEPInheritenceViolationReportBody.h"
#include "CORPViolationReportBody.h"
#include "FrameLoader.h"
#include "HTTPHeaderNames.h"
#include "JSFetchRequestDestination.h"
#include "LocalFrame.h"
#include "PingLoader.h"
#include "RFC8941.h"
#include "Report.h"
#include "ReportingClient.h"
#include "ResourceResponse.h"
#include "ScriptExecutionContext.h"
#include "SecurityOrigin.h"
#include "ViolationReportType.h"
#include <wtf/persistence/PersistentCoders.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

// https://html.spec.whatwg.org/multipage/origin.html#obtain-an-embedder-policy
CrossOriginEmbedderPolicy obtainCrossOriginEmbedderPolicy(const ResourceResponse& response, const ScriptExecutionContext* context)
{
    auto parseCOEPHeader = [&response](HTTPHeaderName headerName, auto& value, auto& reportingEndpoint) {
        auto coepParsingResult = RFC8941::parseItemStructuredFieldValue(response.httpHeaderField(headerName));
        if (!coepParsingResult)
            return;
        auto* policyString = std::get_if<RFC8941::Token>(&coepParsingResult->first);
        if (!policyString || policyString->string() != "require-corp"_s)
            return;

        value = CrossOriginEmbedderPolicyValue::RequireCORP;
        if (auto* reportToString = coepParsingResult->second.getIf<String>("report-to"_s))
            reportingEndpoint = *reportToString;
    };

    CrossOriginEmbedderPolicy policy;
    if (context && !context->settingsValues().crossOriginEmbedderPolicyEnabled)
        return policy;
    if (!SecurityOrigin::create(response.url())->isPotentiallyTrustworthy())
        return policy;

    parseCOEPHeader(HTTPHeaderName::CrossOriginEmbedderPolicy, policy.value, policy.reportingEndpoint);
    parseCOEPHeader(HTTPHeaderName::CrossOriginEmbedderPolicyReportOnly, policy.reportOnlyValue, policy.reportOnlyReportingEndpoint);
    return policy;
}

CrossOriginEmbedderPolicy CrossOriginEmbedderPolicy::isolatedCopy() const &
{
    return {
        value,
        reportOnlyValue,
        reportingEndpoint.isolatedCopy(),
        reportOnlyReportingEndpoint.isolatedCopy()
    };
}

CrossOriginEmbedderPolicy CrossOriginEmbedderPolicy::isolatedCopy() &&
{
    return {
        value,
        reportOnlyValue,
        WTFMove(reportingEndpoint).isolatedCopy(),
        WTFMove(reportOnlyReportingEndpoint).isolatedCopy()
    };
}

void CrossOriginEmbedderPolicy::addPolicyHeadersTo(ResourceResponse& response) const
{
    if (value != CrossOriginEmbedderPolicyValue::UnsafeNone) {
        ASSERT(value == CrossOriginEmbedderPolicyValue::RequireCORP);
        if (reportingEndpoint.isEmpty())
            response.setHTTPHeaderField(HTTPHeaderName::CrossOriginEmbedderPolicy, "require-corp"_s);
        else
            response.setHTTPHeaderField(HTTPHeaderName::CrossOriginEmbedderPolicy, makeString("require-corp; report-to=\""_s, reportingEndpoint, '\"'));
    }
    if (reportOnlyValue != CrossOriginEmbedderPolicyValue::UnsafeNone) {
        ASSERT(reportOnlyValue == CrossOriginEmbedderPolicyValue::RequireCORP);
        if (reportOnlyReportingEndpoint.isEmpty())
            response.setHTTPHeaderField(HTTPHeaderName::CrossOriginEmbedderPolicyReportOnly, "require-corp"_s);
        else
            response.setHTTPHeaderField(HTTPHeaderName::CrossOriginEmbedderPolicyReportOnly, makeString("require-corp; report-to=\""_s, reportOnlyReportingEndpoint, '\"'));
    }
}

// https://html.spec.whatwg.org/multipage/origin.html#queue-a-cross-origin-embedder-policy-inheritance-violation
void sendCOEPInheritenceViolation(ReportingClient& reportingClient, const URL& embedderURL, const String& endpoint, COEPDisposition disposition, const String& type, const URL& blockedURL)
{
    Ref reportBody = COEPInheritenceViolationReportBody::create(disposition, blockedURL, AtomString { type });
    Ref report = Report::create("coep"_s, embedderURL.string(), WTFMove(reportBody));
    reportingClient.notifyReportObservers(WTFMove(report));

    if (endpoint.isEmpty())
        return;

    Ref reportFormData = Report::createReportFormDataForViolation("coep"_s, embedderURL, reportingClient.httpUserAgent(), endpoint, [&](auto& body) {
        body.setString("disposition"_s, disposition == COEPDisposition::Reporting ? "reporting"_s : "enforce"_s);
        body.setString("type"_s, type);
        body.setString("blockedURL"_s, PingLoader::sanitizeURLForReport(blockedURL));
    });
    reportingClient.sendReportToEndpoints(embedderURL, { }, { endpoint }, WTFMove(reportFormData), ViolationReportType::COEPInheritenceViolation);
}

// https://fetch.spec.whatwg.org/#queue-a-cross-origin-embedder-policy-corp-violation-report
void sendCOEPCORPViolation(ReportingClient& reportingClient, const URL& embedderURL, const String& endpoint, COEPDisposition disposition, FetchOptions::Destination destination, const URL& blockedURL)
{
    Ref reportBody = CORPViolationReportBody::create(disposition, blockedURL, destination);
    Ref report = Report::create("coep"_s, embedderURL.string(), WTFMove(reportBody));
    reportingClient.notifyReportObservers(WTFMove(report));

    if (endpoint.isEmpty())
        return;

    Ref reportFormData = Report::createReportFormDataForViolation("coep"_s, embedderURL, reportingClient.httpUserAgent(), endpoint, [&](auto& body) {
        body.setString("disposition"_s, disposition == COEPDisposition::Reporting ? "reporting"_s : "enforce"_s);
        body.setString("type"_s, "corp"_s);
        body.setString("blockedURL"_s, PingLoader::sanitizeURLForReport(blockedURL));
        body.setString("destination"_s, convertEnumerationToString(destination));
    });
    reportingClient.sendReportToEndpoints(embedderURL, { }, { endpoint }, WTFMove(reportFormData), ViolationReportType::CORPViolation);
}

void CrossOriginEmbedderPolicy::encode(WTF::Persistence::Encoder& encoder) const
{
    encoder << value << reportingEndpoint << reportOnlyValue << reportOnlyReportingEndpoint;
}

std::optional<CrossOriginEmbedderPolicy> CrossOriginEmbedderPolicy::decode(WTF::Persistence::Decoder& decoder)
{
    std::optional<CrossOriginEmbedderPolicyValue> value;
    decoder >> value;
    if (!value)
        return std::nullopt;

    std::optional<String> reportingEndpoint;
    decoder >> reportingEndpoint;
    if (!reportingEndpoint)
        return std::nullopt;

    std::optional<CrossOriginEmbedderPolicyValue> reportOnlyValue;
    decoder >> reportOnlyValue;
    if (!reportOnlyValue)
        return std::nullopt;

    std::optional<String> reportOnlyReportingEndpoint;
    decoder >> reportOnlyReportingEndpoint;
    if (!reportOnlyReportingEndpoint)
        return std::nullopt;

    return { {
        *value,
        *reportOnlyValue,
        WTFMove(*reportingEndpoint),
        WTFMove(*reportOnlyReportingEndpoint)
    } };
}

} // namespace WebCore
