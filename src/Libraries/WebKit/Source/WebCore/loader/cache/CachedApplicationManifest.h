/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
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

#if ENABLE(APPLICATION_MANIFEST)

#include "ApplicationManifest.h"
#include "CachedResource.h"

namespace WebCore {

class Document;
class TextResourceDecoder;

class CachedApplicationManifest final : public CachedResource {
public:
    CachedApplicationManifest(CachedResourceRequest&&, PAL::SessionID, const CookieJar*);

    std::optional<struct ApplicationManifest> process(const URL& manifestURL, const URL& documentURL, Document* = nullptr);

private:
    void finishLoading(const FragmentedSharedBuffer*, const NetworkLoadMetrics&) final;
    const TextResourceDecoder* textResourceDecoder() const final { return m_decoder.ptr(); }
    Ref<TextResourceDecoder> protectedDecoder() const;
    void setEncoding(const String&) final;
    ASCIILiteral encoding() const final;

    Ref<TextResourceDecoder> m_decoder;
    std::optional<String> m_text;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE(CachedApplicationManifest, CachedResource::Type::ApplicationManifest)

#endif // ENABLE(APPLICATION_MANIFEST)
