/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

#include "CrossOriginEmbedderPolicy.h"
#include "ReportBody.h"
#include "ViolationReportType.h"
#include <wtf/ArgumentCoder.h>

namespace WebCore {

class COEPInheritenceViolationReportBody : public ReportBody {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(COEPInheritenceViolationReportBody);
public:
    WEBCORE_EXPORT static Ref<COEPInheritenceViolationReportBody> create(COEPDisposition, const URL& blockedURL, const String& type);

    String disposition() const;
    const String& type() const final { return m_type; }
    const String& blockedURL() const { return m_blockedURL.string(); }

private:
    friend struct IPC::ArgumentCoder<COEPInheritenceViolationReportBody, void>;
    COEPInheritenceViolationReportBody(COEPDisposition, const URL& blockedURL, const String& type);

    ViolationReportType reportBodyType() const final { return ViolationReportType::COEPInheritenceViolation; }

    COEPDisposition m_disposition;
    URL m_blockedURL;
    String m_type;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::COEPInheritenceViolationReportBody)
    static bool isType(const WebCore::ReportBody& reportBody) { return reportBody.reportBodyType() == WebCore::ViolationReportType::COEPInheritenceViolation; }
SPECIALIZE_TYPE_TRAITS_END()
