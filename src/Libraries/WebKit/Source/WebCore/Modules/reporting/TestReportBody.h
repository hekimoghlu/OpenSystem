/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#include <wtf/text/WTFString.h>

namespace WebCore {

class FormData;

class TestReportBody final : public ReportBody {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TestReportBody);
public:
    WEBCORE_EXPORT static Ref<TestReportBody> create(String&& message);

    WEBCORE_EXPORT const String& type() const final;
    WEBCORE_EXPORT const String& message() const;

private:
    TestReportBody(String&& message);

    ViolationReportType reportBodyType() const final { return ViolationReportType::Test; }

    const String m_bodyMessage;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::TestReportBody)
    static bool isType(const WebCore::ReportBody& reportBody) { return reportBody.reportBodyType() == WebCore::ViolationReportType::Test; }
SPECIALIZE_TYPE_TRAITS_END()
