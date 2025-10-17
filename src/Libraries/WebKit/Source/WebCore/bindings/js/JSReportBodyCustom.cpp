/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
#include "JSReportBody.h"

#include "COEPInheritenceViolationReportBody.h"
#include "CORPViolationReportBody.h"
#include "CSPViolationReportBody.h"
#include "DeprecationReportBody.h"
#include "JSCOEPInheritenceViolationReportBody.h"
#include "JSCORPViolationReportBody.h"
#include "JSCSPViolationReportBody.h"
#include "JSDOMBinding.h"
#include "JSDeprecationReportBody.h"
#include "JSTestReportBody.h"
#include "ReportBody.h"
#include "TestReportBody.h"
#include "ViolationReportType.h"

namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<ReportBody>&& reportBody)
{
    if (is<CSPViolationReportBody>(reportBody))
        return createWrapper<CSPViolationReportBody>(globalObject, WTFMove(reportBody));
    if (is<COEPInheritenceViolationReportBody>(reportBody))
        return createWrapper<COEPInheritenceViolationReportBody>(globalObject, WTFMove(reportBody));
    if (is<CORPViolationReportBody>(reportBody))
        return createWrapper<CORPViolationReportBody>(globalObject, WTFMove(reportBody));
    if (is<DeprecationReportBody>(reportBody))
        return createWrapper<DeprecationReportBody>(globalObject, WTFMove(reportBody));
    if (is<TestReportBody>(reportBody))
        return createWrapper<TestReportBody>(globalObject, WTFMove(reportBody));
    return createWrapper<ReportBody>(globalObject, WTFMove(reportBody));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, ReportBody& reportBody)
{
    return wrap(lexicalGlobalObject, globalObject, reportBody);
}

JSValue JSReportBody::toJSON(JSGlobalObject& lexicalGlobalObject, CallFrame& callFrame)
{
    UNUSED_PARAM(lexicalGlobalObject);
    UNUSED_PARAM(callFrame);

    return jsUndefined();
}

} // namepace WebCore
