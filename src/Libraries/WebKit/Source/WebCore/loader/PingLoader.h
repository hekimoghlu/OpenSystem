/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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

#include "ReferrerPolicy.h"
#include "SecurityOriginData.h"
#include <wtf/Forward.h>
#include <wtf/Ref.h>

namespace WebCore {

class FormData;
class HTTPHeaderMap;
class LocalFrame;
class ResourceRequest;

enum class ContentSecurityPolicyImposition : uint8_t;
enum class ViolationReportType : uint8_t;

class PingLoader {
public:
    static void loadImage(LocalFrame&, const URL&);
    static void sendPing(LocalFrame&, const URL& pingURL, const URL& destinationURL);
    WEBCORE_EXPORT static void sendViolationReport(LocalFrame&, const URL& reportURL, Ref<FormData>&& report, ViolationReportType);

    static String sanitizeURLForReport(const URL&);

private:
    enum class ShouldFollowRedirects : bool { No, Yes };
    static void startPingLoad(LocalFrame&, ResourceRequest&, HTTPHeaderMap&& originalRequestHeaders, ShouldFollowRedirects, ContentSecurityPolicyImposition, ReferrerPolicy, std::optional<ViolationReportType> = std::nullopt);
};

} // namespace WebCore
