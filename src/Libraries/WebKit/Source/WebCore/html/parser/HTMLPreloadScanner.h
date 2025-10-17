/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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

#include "CSSPreloadScanner.h"
#include "HTMLTokenizer.h"
#include "SegmentedString.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class TokenPreloadScanner {
    WTF_MAKE_TZONE_ALLOCATED(TokenPreloadScanner);
    WTF_MAKE_NONCOPYABLE(TokenPreloadScanner);
public:
    explicit TokenPreloadScanner(const URL& documentURL, float deviceScaleFactor = 1.0);

    void scan(const HTMLToken&, PreloadRequestStream&, Document&);

    void setPredictedBaseElementURL(const URL& url) { m_predictedBaseElementURL = url; }
    
    bool inPicture() { return !m_pictureSourceState.isEmpty(); }

private:
    enum class TagId {
        // These tags are scanned by the StartTagScanner.
        Img,
        Input,
        Link,
        Script,
        Meta,
        Source,
        Video,

        // These tags are not scanned by the StartTagScanner.
        Unknown,
        Style,
        Base,
        Template,
        Picture
    };

    class StartTagScanner;

    static TagId tagIdFor(const HTMLToken::DataVector&);

    static ASCIILiteral initiatorFor(TagId);

    void updatePredictedBaseURL(const HTMLToken&, bool shouldRestrictBaseURLSchemes);

    CSSPreloadScanner m_cssScanner;
    const URL m_documentURL;
    const float m_deviceScaleFactor { 1 };

    URL m_predictedBaseElementURL;
    bool m_inStyle { false };
    
    Vector<bool> m_pictureSourceState;

    unsigned m_templateCount { 0 };
};

class HTMLPreloadScanner {
    WTF_MAKE_TZONE_ALLOCATED(HTMLPreloadScanner);
public:
    HTMLPreloadScanner(const HTMLParserOptions&, const URL& documentURL, float deviceScaleFactor = 1.0);

    void appendToEnd(const SegmentedString&);
    void scan(HTMLResourcePreloader&, Document&);

private:
    TokenPreloadScanner m_scanner;
    SegmentedString m_source;
    HTMLTokenizer m_tokenizer;
};

WEBCORE_EXPORT bool testPreloadScannerViewportSupport(Document*);

} // namespace WebCore
