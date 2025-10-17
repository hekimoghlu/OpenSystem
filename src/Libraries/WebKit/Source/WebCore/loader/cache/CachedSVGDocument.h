/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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
#include "SVGDocument.h"
#include "TextResourceDecoder.h"

namespace WebCore {

class Settings;

class CachedSVGDocument final : public CachedResource {
public:
    explicit CachedSVGDocument(CachedResourceRequest&&, PAL::SessionID, const CookieJar*, const Settings&);
    explicit CachedSVGDocument(CachedResourceRequest&&, CachedSVGDocument&);
    virtual ~CachedSVGDocument();

    SVGDocument* document() const { return m_document.get(); }

private:
    bool mayTryReplaceEncodedData() const override { return true; }
    void setEncoding(const String&) override;
    ASCIILiteral encoding() const override;
    const TextResourceDecoder* textResourceDecoder() const override { return m_decoder.get(); }
    RefPtr<TextResourceDecoder> protectedDecoder() const;
    void finishLoading(const FragmentedSharedBuffer*, const NetworkLoadMetrics&) override;

    RefPtr<SVGDocument> m_document;
    RefPtr<TextResourceDecoder> m_decoder;
    const Ref<const Settings> m_settings;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE(CachedSVGDocument, CachedResource::Type::SVGDocumentResource)
