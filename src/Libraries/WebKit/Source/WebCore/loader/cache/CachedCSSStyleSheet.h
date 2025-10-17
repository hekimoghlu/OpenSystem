/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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

#include "CachedResource.h"
#include "FrameLoaderTypes.h"

namespace WebCore {

class FrameLoader;
class StyleSheetContents;
class TextResourceDecoder;

struct CSSParserContext;

class CachedCSSStyleSheet final : public CachedResource {
public:
    CachedCSSStyleSheet(CachedResourceRequest&&, PAL::SessionID, const CookieJar*);
    virtual ~CachedCSSStyleSheet();

    enum class MIMETypeCheckHint { Strict, Lax };
    const String sheetText(MIMETypeCheckHint = MIMETypeCheckHint::Strict, bool* hasValidMIMEType = nullptr, bool* hasHTTPStatusOK = nullptr) const;

    RefPtr<StyleSheetContents> restoreParsedStyleSheet(const CSSParserContext&, CachePolicy, FrameLoader&);
    void saveParsedStyleSheet(Ref<StyleSheetContents>&&);

    bool mimeTypeAllowedByNosniff() const;

private:
    String responseMIMEType() const;
    bool canUseSheet(MIMETypeCheckHint, bool* hasValidMIMEType, bool* hasHTTPStatusOK) const;
    bool mayTryReplaceEncodedData() const final { return true; }
    Ref<TextResourceDecoder> protectedDecoder() const;

    void didAddClient(CachedResourceClient&) final;

    void setEncoding(const String&) final;
    ASCIILiteral encoding() const final;
    const TextResourceDecoder* textResourceDecoder() const final { return m_decoder.ptr(); }
    void finishLoading(const FragmentedSharedBuffer*, const NetworkLoadMetrics&) final;
    void destroyDecodedData() final;

    void setBodyDataFrom(const CachedResource&) final;

    void checkNotify(const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess = LoadWillContinueInAnotherProcess::No) final;

    Ref<TextResourceDecoder> m_decoder;
    String m_decodedSheetText;

    RefPtr<StyleSheetContents> m_parsedStyleSheetCache;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE(CachedCSSStyleSheet, CachedResource::Type::CSSStyleSheet)
