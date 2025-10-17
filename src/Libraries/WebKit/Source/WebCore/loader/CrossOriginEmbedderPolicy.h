/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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

#include "FetchOptions.h"
#include <wtf/text/WTFString.h>

namespace WTF::Persistence {

class Decoder;
class Encoder;

}

namespace WebCore {

class LocalFrame;
class ResourceResponse;
class ScriptExecutionContext;

struct ReportingClient;
class SecurityOriginData;

// https://html.spec.whatwg.org/multipage/origin.html#embedder-policy-value
enum class CrossOriginEmbedderPolicyValue : bool {
    UnsafeNone,
    RequireCORP
};

// https://html.spec.whatwg.org/multipage/origin.html#embedder-policy
struct CrossOriginEmbedderPolicy {
    CrossOriginEmbedderPolicyValue value { CrossOriginEmbedderPolicyValue::UnsafeNone };
    CrossOriginEmbedderPolicyValue reportOnlyValue { CrossOriginEmbedderPolicyValue::UnsafeNone };
    String reportingEndpoint;
    String reportOnlyReportingEndpoint;

    CrossOriginEmbedderPolicy isolatedCopy() const &;
    CrossOriginEmbedderPolicy isolatedCopy() &&;
    void encode(WTF::Persistence::Encoder&) const;
    static std::optional<CrossOriginEmbedderPolicy> decode(WTF::Persistence::Decoder &);

    friend bool operator==(const CrossOriginEmbedderPolicy&, const CrossOriginEmbedderPolicy&) = default;

    void addPolicyHeadersTo(ResourceResponse&) const;
};

enum class COEPDisposition : bool { Reporting , Enforce };

WEBCORE_EXPORT CrossOriginEmbedderPolicy obtainCrossOriginEmbedderPolicy(const ResourceResponse&, const ScriptExecutionContext*);
WEBCORE_EXPORT void sendCOEPInheritenceViolation(ReportingClient&, const URL& embedderURL, const String& endpoint, COEPDisposition, const String& type, const URL& blockedURL);
WEBCORE_EXPORT void sendCOEPCORPViolation(ReportingClient&, const URL& embedderURL, const String& endpoint, COEPDisposition, FetchOptions::Destination, const URL& blockedURL);

} // namespace WebCore
