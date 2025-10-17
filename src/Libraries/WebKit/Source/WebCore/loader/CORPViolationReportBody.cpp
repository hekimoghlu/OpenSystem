/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
#include "CORPViolationReportBody.h"

#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CORPViolationReportBody);

Ref<CORPViolationReportBody> CORPViolationReportBody::create(COEPDisposition disposition, const URL& blockedURL, FetchOptions::Destination destination)
{
    return adoptRef(*new CORPViolationReportBody(disposition, blockedURL, destination));
}

CORPViolationReportBody::CORPViolationReportBody(COEPDisposition disposition, const URL& blockedURL, FetchOptions::Destination destination)
    : m_disposition(disposition)
    , m_blockedURL(blockedURL)
    , m_destination(destination)
{
}

const String& CORPViolationReportBody::type() const
{
    static NeverDestroyed<const String> corpType(MAKE_STATIC_STRING_IMPL("corp"));
    return corpType;
}

String CORPViolationReportBody::disposition() const
{
    return m_disposition == COEPDisposition::Reporting ? "reporting"_s : "enforce"_s;
}

} // namespace WebCore
