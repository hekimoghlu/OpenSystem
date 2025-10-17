/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#include "ViolationReportType.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WallTime.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class FormData;

class DeprecationReportBody final : public ReportBody {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DeprecationReportBody);
public:
    WEBCORE_EXPORT static Ref<DeprecationReportBody> create(String&& id, WallTime anticipatedRemoval, String&& message, String&& sourceFile, std::optional<unsigned> lineNumber, std::optional<unsigned> columnNumber);

    const String& type() const final;
    const String& id() const { return m_id; };
    WallTime anticipatedRemoval() const { return m_anticipatedRemoval; }
    const String& message() const { return m_message; }
    const String& sourceFile() const { return m_sourceFile; }
    std::optional<unsigned> lineNumber() const { return m_lineNumber; }
    std::optional<unsigned> columnNumber() const { return m_columnNumber; }

    WEBCORE_EXPORT Ref<FormData> createReportFormDataForViolation() const;

private:
    DeprecationReportBody(String&& id, WallTime anticipatedRemoval, String&& message, String&& sourceFile, std::optional<unsigned> lineNumber, std::optional<unsigned> columnNumber);

    ViolationReportType reportBodyType() const final { return ViolationReportType::Deprecation; }

    const String m_id;
    const WallTime m_anticipatedRemoval;
    const String m_message;
    const String m_sourceFile;
    const std::optional<unsigned> m_lineNumber;
    const std::optional<unsigned> m_columnNumber;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::DeprecationReportBody)
    static bool isType(const WebCore::ReportBody& reportBody) { return reportBody.reportBodyType() == WebCore::ViolationReportType::Deprecation; }
SPECIALIZE_TYPE_TRAITS_END()
