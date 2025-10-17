/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

namespace WebCore {

class TextResourceDecoder;

enum class ScriptRequiresTelemetry : bool { No, Yes };

class CachedScript final : public CachedResource {
public:
    CachedScript(CachedResourceRequest&&, PAL::SessionID, const CookieJar*, ScriptRequiresTelemetry);
    virtual ~CachedScript();

    enum class ShouldDecodeAsUTF8Only : bool { No, Yes };
    WEBCORE_EXPORT StringView script(ShouldDecodeAsUTF8Only = ShouldDecodeAsUTF8Only::No);
    WEBCORE_EXPORT unsigned scriptHash(ShouldDecodeAsUTF8Only = ShouldDecodeAsUTF8Only::No);

    bool requiresTelemetry() const { return m_requiresTelemetry; }

private:
    bool mayTryReplaceEncodedData() const final { return true; }

    void setEncoding(const String&) final;
    ASCIILiteral encoding() const final;
    const TextResourceDecoder* textResourceDecoder() const final { return m_decoder.get(); }
    RefPtr<TextResourceDecoder> protectedDecoder() const;
    void finishLoading(const FragmentedSharedBuffer*, const NetworkLoadMetrics&) final;

    void destroyDecodedData() final;

    void setBodyDataFrom(const CachedResource&) final;

    String m_script;
    unsigned m_scriptHash { 0 };
    bool m_wasForceDecodedAsUTF8 { false };
    bool m_requiresTelemetry { false };

    enum DecodingState { NeverDecoded, DataAndDecodedStringHaveSameBytes, DataAndDecodedStringHaveDifferentBytes };
    DecodingState m_decodingState { NeverDecoded };

    RefPtr<TextResourceDecoder> m_decoder;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE(CachedScript, CachedResource::Type::Script)
