/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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
#include "TestReportBody.h"

#include "FormData.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TestReportBody);

TestReportBody::TestReportBody(String&& message)
    : m_bodyMessage(WTFMove(message))
{
}

Ref<TestReportBody> TestReportBody::create(String&& message)
{
    return adoptRef(*new TestReportBody(WTFMove(message)));
}

const String& TestReportBody::type() const
{
    static NeverDestroyed<const String> testReportType(MAKE_STATIC_STRING_IMPL("test"));
    return testReportType;
}

const String& TestReportBody::message() const
{
    // https://w3c.github.io/reporting/#generate-test-report-command, Step 7.1.7
    return m_bodyMessage;
}

} // namespace WebCore
