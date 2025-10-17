/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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

#include <wtf/JSONValues.h>
#include "ReportBody.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class FormData;

class WEBCORE_EXPORT Report : public RefCounted<Report> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(Report, WEBCORE_EXPORT);
public:
    static Ref<Report> create(const String& type, const String& url, RefPtr<ReportBody>&&);

    ~Report();

    const String& type() const;
    const String& url() const;
    const RefPtr<ReportBody>& body() const;

    static Ref<FormData> createReportFormDataForViolation(const String& type, const URL&, const String& userAgent, const String& destination, const Function<void(JSON::Object&)>& populateBody);

private:
    explicit Report(const String& type, const String& url, RefPtr<ReportBody>&&);

    String m_type;
    String m_url;
    RefPtr<ReportBody> m_body;
};

} // namespace WebCore
