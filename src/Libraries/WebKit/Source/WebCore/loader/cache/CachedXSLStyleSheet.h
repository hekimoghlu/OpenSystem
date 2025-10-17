/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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

#if ENABLE(XSLT)

#include "CachedResource.h"

namespace WebCore {

class TextResourceDecoder;

class CachedXSLStyleSheet final : public CachedResource {
public:
    CachedXSLStyleSheet(CachedResourceRequest&&, PAL::SessionID, const CookieJar*);
    virtual ~CachedXSLStyleSheet();

    const String& sheet() const { return m_sheet; }

private:
    void checkNotify(const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess = LoadWillContinueInAnotherProcess::No) final;
    bool mayTryReplaceEncodedData() const final { return true; }
    void didAddClient(CachedResourceClient&) final;
    void setEncoding(const String&) final;
    ASCIILiteral encoding() const final;
    const TextResourceDecoder* textResourceDecoder() const final { return m_decoder.get(); }
    RefPtr<TextResourceDecoder> protectedDecoder() const;
    void finishLoading(const FragmentedSharedBuffer*, const NetworkLoadMetrics&) final;

    String m_sheet;
    RefPtr<TextResourceDecoder> m_decoder;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE(CachedXSLStyleSheet, CachedResource::Type::XSLStyleSheet)

#endif // ENABLE(XSLT)
