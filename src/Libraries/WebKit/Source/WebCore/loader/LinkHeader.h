/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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

#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class LinkHeader {
public:
    template<typename CharacterType> LinkHeader(StringParsingBuffer<CharacterType>&);

    const String& url() const { return m_url; }
    const String& rel() const { return m_rel; }
    const String& as() const { return m_as; }
    const String& mimeType() const { return m_mimeType; }
    const String& media() const { return m_media; }
    const String& crossOrigin() const { return m_crossOrigin; }
    const String& imageSrcSet() const { return m_imageSrcSet; }
    const String& imageSizes() const { return m_imageSizes; }
    const String& nonce() const { return m_nonce; }
    const String& referrerPolicy() const { return m_referrerPolicy; }
    const String& fetchPriority() const { return m_fetchPriority; }
    bool valid() const { return m_isValid; }
    bool isViewportDependent() const { return !media().isEmpty() || !imageSrcSet().isEmpty() || !imageSizes().isEmpty(); }

    enum LinkParameterName {
        LinkParameterRel,
        LinkParameterAnchor,
        LinkParameterTitle,
        LinkParameterMedia,
        LinkParameterType,
        LinkParameterRev,
        LinkParameterHreflang,
        // Beyond this point, only link-extension parameters
        LinkParameterUnknown,
        LinkParameterCrossOrigin,
        LinkParameterAs,
        LinkParameterImageSrcSet,
        LinkParameterImageSizes,
        LinkParameterNonce,
        LinkParameterReferrerPolicy,
        LinkParameterFetchPriority,
    };

private:
    void setValue(LinkParameterName, String&& value);

    String m_url;
    String m_rel;
    String m_as;
    String m_mimeType;
    String m_media;
    String m_crossOrigin;
    String m_imageSrcSet;
    String m_imageSizes;
    String m_nonce;
    String m_referrerPolicy;
    String m_fetchPriority;
    bool m_isValid { true };
};

class LinkHeaderSet {
public:
    WEBCORE_EXPORT LinkHeaderSet(const String& header);

    Vector<LinkHeader>::const_iterator begin() const { return m_headerSet.begin(); }
    Vector<LinkHeader>::const_iterator end() const { return m_headerSet.end(); }

private:
    Vector<LinkHeader> m_headerSet;
};

} // namespace WebCore

