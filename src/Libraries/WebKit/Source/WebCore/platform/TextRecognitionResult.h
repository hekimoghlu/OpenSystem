/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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

#if ENABLE(IMAGE_ANALYSIS)

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
OBJC_CLASS NSAttributedString;
OBJC_CLASS NSData;
OBJC_CLASS VKCImageAnalysis;
#endif

#if ENABLE(DATA_DETECTION)
OBJC_CLASS DDScannerResult;
#endif

#include "FloatQuad.h"
#include <wtf/RetainPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct CharacterRange;

struct TextRecognitionWordData {
    TextRecognitionWordData(const String& theText, FloatQuad&& quad, bool leadingWhitespace)
        : text(theText)
        , normalizedQuad(WTFMove(quad))
        , hasLeadingWhitespace(leadingWhitespace)
    {
    }

    String text;
    FloatQuad normalizedQuad;
    bool hasLeadingWhitespace { true };
};

struct TextRecognitionLineData {
    TextRecognitionLineData(FloatQuad&& quad, Vector<TextRecognitionWordData>&& theChildren, bool newline, bool isVertical)
        : normalizedQuad(WTFMove(quad))
        , children(WTFMove(theChildren))
        , hasTrailingNewline(newline)
        , isVertical(isVertical)
    {
    }

    FloatQuad normalizedQuad;
    Vector<TextRecognitionWordData> children;
    bool hasTrailingNewline { true };
    bool isVertical { false };
};

#if ENABLE(DATA_DETECTION)

struct TextRecognitionDataDetector {
    TextRecognitionDataDetector() = default;
    TextRecognitionDataDetector(RetainPtr<DDScannerResult>&& scannerResult, Vector<FloatQuad>&& quads)
        : result(WTFMove(scannerResult))
        , normalizedQuads(WTFMove(quads))
    {
    }

    RetainPtr<DDScannerResult> result;
    Vector<FloatQuad> normalizedQuads;
};

#endif // ENABLE(DATA_DETECTION)

struct TextRecognitionBlockData {
    TextRecognitionBlockData(const String& theText, FloatQuad&& quad)
        : text(theText)
        , normalizedQuad(WTFMove(quad))
    {
    }

    String text;
    FloatQuad normalizedQuad;
};

struct TextRecognitionResult {
    Vector<TextRecognitionLineData> lines;

#if ENABLE(DATA_DETECTION)
    Vector<TextRecognitionDataDetector> dataDetectors;
#endif

    Vector<TextRecognitionBlockData> blocks;

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
    RetainPtr<NSData> imageAnalysisData;

    WEBCORE_EXPORT static RetainPtr<NSData> encodeVKCImageAnalysis(RetainPtr<VKCImageAnalysis>);
    WEBCORE_EXPORT static RetainPtr<VKCImageAnalysis> decodeVKCImageAnalysis(RetainPtr<NSData>);
#endif

    bool isEmpty() const
    {
        if (!lines.isEmpty())
            return false;

#if ENABLE(DATA_DETECTION)
        if (!dataDetectors.isEmpty())
            return false;
#endif

        if (!blocks.isEmpty())
            return false;

        return true;
    }
};

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
RetainPtr<NSAttributedString> stringForRange(const TextRecognitionResult&, const CharacterRange&);
#endif

} // namespace WebCore

#endif // ENABLE(IMAGE_ANALYSIS)
