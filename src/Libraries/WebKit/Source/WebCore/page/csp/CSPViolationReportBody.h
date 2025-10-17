/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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

#include "ReportBody.h"
#include "SecurityPolicyViolationEvent.h"
#include "ViolationReportType.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FormData;

struct CSPInfo;
struct SecurityPolicyViolationEventInit;

class CSPViolationReportBody final : public ReportBody {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSPViolationReportBody);
public:
    using Init = SecurityPolicyViolationEventInit;

    WEBCORE_EXPORT static Ref<CSPViolationReportBody> create(Init&&);
    WEBCORE_EXPORT static Ref<CSPViolationReportBody> create(String&& documentURL, String&& referrer, String&& blockedURL, String&& effectiveDirective, String&& originalPolicy, String&& sourceFile, String&& sample, SecurityPolicyViolationEventDisposition, unsigned short statusCode, unsigned long lineNumber, unsigned long columnNumber);

    const String& type() const final;
    const String& documentURL() const { return m_documentURL; }
    const String& referrer() const { return m_referrer; }
    const String& blockedURL() const { return m_blockedURL; }
    const String& effectiveDirective() const { return m_effectiveDirective; }
    const String& originalPolicy() const { return m_originalPolicy; }
    const String& sourceFile() const { return m_sourceFile; }
    const String& sample() const { return m_sample; }
    SecurityPolicyViolationEventDisposition disposition() const { return m_disposition; }
    unsigned short statusCode() const { return m_statusCode; }
    unsigned long lineNumber() const { return m_lineNumber; }
    unsigned long columnNumber() const { return m_columnNumber; }
    
    WEBCORE_EXPORT Ref<FormData> createReportFormDataForViolation(bool usesReportTo, bool isReportOnly) const;

private:
    CSPViolationReportBody(Init&&);
    CSPViolationReportBody(String&& documentURL, String&& referrer, String&& blockedURL, String&& effectiveDirective, String&& originalPolicy, String&& sourceFile, String&& sample, SecurityPolicyViolationEventDisposition, unsigned short statusCode, unsigned long lineNumber, unsigned long columnNumber);

    ViolationReportType reportBodyType() const final { return ViolationReportType::ContentSecurityPolicy; }

    const String m_documentURL;
    const String m_referrer;
    const String m_blockedURL;
    const String m_effectiveDirective;
    const String m_originalPolicy;
    const String m_sourceFile;
    const String m_sample;
    const SecurityPolicyViolationEventDisposition m_disposition;
    const unsigned short m_statusCode;
    const unsigned long m_lineNumber;
    const unsigned long m_columnNumber;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSPViolationReportBody)
    static bool isType(const WebCore::ReportBody& reportBody) { return reportBody.reportBodyType() == WebCore::ViolationReportType::ContentSecurityPolicy; }
SPECIALIZE_TYPE_TRAITS_END()
