/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

#include <optional>

namespace WebCore {

#if ENABLE(APP_HIGHLIGHTS)
class AppHighlightRangeData;
#endif
class CertificateInfo;
class ContentSecurityPolicyResponseHeaders;
class HTTPHeaderMap;
class ResourceResponse;
class ResourceRequest;

struct ClientOrigin;
struct CrossOriginEmbedderPolicy;
struct FetchOptions;
struct ImageResource;
struct ImportedScriptAttributes;
struct NavigationPreloadState;
class SecurityOriginData;

}

namespace WTF::Persistence {

template<typename> struct Coder;
class Decoder;
class Encoder;

#define DECLARE_CODER(class) \
template<> struct Coder<class> { \
    WEBCORE_EXPORT static void encodeForPersistence(Encoder&, const class&); \
    WEBCORE_EXPORT static std::optional<class> decodeForPersistence(Decoder&); \
}

#if ENABLE(APP_HIGHLIGHTS)
DECLARE_CODER(WebCore::AppHighlightRangeData);
#endif
DECLARE_CODER(WebCore::CertificateInfo);
DECLARE_CODER(WebCore::ClientOrigin);
DECLARE_CODER(WebCore::ContentSecurityPolicyResponseHeaders);
DECLARE_CODER(WebCore::CrossOriginEmbedderPolicy);
DECLARE_CODER(WebCore::FetchOptions);
DECLARE_CODER(WebCore::HTTPHeaderMap);
DECLARE_CODER(WebCore::ImportedScriptAttributes);
DECLARE_CODER(WebCore::ImageResource);
DECLARE_CODER(WebCore::ResourceResponse);
DECLARE_CODER(WebCore::ResourceRequest);
DECLARE_CODER(WebCore::SecurityOriginData);
DECLARE_CODER(WebCore::NavigationPreloadState);
#undef DECLARE_CODER

}
