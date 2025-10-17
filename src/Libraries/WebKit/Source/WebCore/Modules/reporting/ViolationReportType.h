/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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

namespace WebCore {

enum class ViolationReportType : uint8_t {
    COEPInheritenceViolation, // https://html.spec.whatwg.org/multipage/origin.html#queue-a-cross-origin-embedder-policy-inheritance-violation
    CORPViolation, // https://fetch.spec.whatwg.org/#queue-a-cross-origin-embedder-policy-corp-violation-report
    CSPHashReport,
    ContentSecurityPolicy,
    CrossOriginOpenerPolicy,
    Deprecation, // https://wicg.github.io/deprecation-reporting/
    StandardReportingAPIViolation, // https://www.w3.org/TR/reporting/#try-delivery
    Test, // https://www.w3.org/TR/reporting-1/#generate-test-report-command
    // More to come
};

} // namespace WebCore
