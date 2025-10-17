/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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

#include "HTTPHeaderNames.h"
#include "ReferrerPolicy.h"
#include "ResourceResponse.h"
#include "StoredCredentialsPolicy.h"
#include <wtf/Expected.h>
#include <wtf/Forward.h>
#include <wtf/OptionSet.h>

namespace PAL {
class SessionID;
}

namespace WebCore {

class CachedResourceRequest;
class Document;
class HTTPHeaderMap;
class OriginAccessPatterns;
class ResourceError;
class ResourceRequest;
class ResourceResponse;
class SecurityOrigin;

struct ResourceLoaderOptions;

enum class CrossOriginEmbedderPolicyValue : bool;

WEBCORE_EXPORT bool isSimpleCrossOriginAccessRequest(const String& method, const HTTPHeaderMap&);
bool isOnAccessControlSimpleRequestMethodAllowlist(const String&);

void updateRequestReferrer(ResourceRequest&, ReferrerPolicy, const URL&, const OriginAccessPatterns&);
    
WEBCORE_EXPORT void updateRequestForAccessControl(ResourceRequest&, SecurityOrigin&, StoredCredentialsPolicy);

WEBCORE_EXPORT ResourceRequest createAccessControlPreflightRequest(const ResourceRequest&, SecurityOrigin&, const String&, bool includeFetchMetadata);
enum class SameOriginFlag : bool { No, Yes };
CachedResourceRequest createPotentialAccessControlRequest(ResourceRequest&&, ResourceLoaderOptions&&, Document&, const String& crossOriginAttribute, SameOriginFlag = SameOriginFlag::No);

enum class HTTPHeadersToKeepFromCleaning : uint8_t {
    ContentType = 1 << 0,
    Referer = 1 << 1,
    Origin = 1 << 2,
    UserAgent = 1 << 3,
    AcceptEncoding = 1 << 4,
    CacheControl = 1 << 5,
    Pragma = 1 << 6
};

OptionSet<HTTPHeadersToKeepFromCleaning> httpHeadersToKeepFromCleaning(const HTTPHeaderMap&);
WEBCORE_EXPORT void cleanHTTPRequestHeadersForAccessControl(ResourceRequest&, OptionSet<HTTPHeadersToKeepFromCleaning>);

class WEBCORE_EXPORT CrossOriginAccessControlCheckDisabler {
public:
    static CrossOriginAccessControlCheckDisabler& singleton();
    virtual ~CrossOriginAccessControlCheckDisabler() = default;
    void setCrossOriginAccessControlCheckEnabled(bool);
    virtual bool crossOriginAccessControlCheckEnabled() const;
private:
    bool m_accessControlCheckEnabled { true };
};

WEBCORE_EXPORT Expected<void, String> passesAccessControlCheck(const ResourceResponse&, StoredCredentialsPolicy, const SecurityOrigin&, const CrossOriginAccessControlCheckDisabler*);
WEBCORE_EXPORT Expected<void, String> validatePreflightResponse(PAL::SessionID, const ResourceRequest&, const ResourceResponse&, StoredCredentialsPolicy, const SecurityOrigin& topOrigin, const SecurityOrigin& securityOrigin, const CrossOriginAccessControlCheckDisabler*);

enum class ForNavigation : bool { No, Yes };
WEBCORE_EXPORT std::optional<ResourceError> validateCrossOriginResourcePolicy(CrossOriginEmbedderPolicyValue, const SecurityOrigin&, const URL&, const ResourceResponse&, ForNavigation, const OriginAccessPatterns&);
WEBCORE_EXPORT std::optional<ResourceError> validateCrossOriginResourcePolicy(CrossOriginEmbedderPolicyValue, const SecurityOrigin&, const URL&, bool isResponseNull, const URL& responseURL, const String& crossOriginResourcePolicyHeaderValue, ForNavigation, const OriginAccessPatterns&);
std::optional<ResourceError> validateRangeRequestedFlag(const ResourceRequest&, const ResourceResponse&);
String validateCrossOriginRedirectionURL(const URL&);

} // namespace WebCore
