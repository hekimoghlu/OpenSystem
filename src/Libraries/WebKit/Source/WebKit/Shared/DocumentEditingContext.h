/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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

#if PLATFORM(IOS_FAMILY)

#include "ArgumentCoders.h"
#include "WKBrowserEngineDefinitions.h"
#include <WebCore/AttributedString.h>
#include <WebCore/ElementContext.h>
#include <WebCore/FloatRect.h>
#include <WebCore/TextGranularity.h>
#include <wtf/OptionSet.h>
#include <wtf/Vector.h>

OBJC_CLASS WKBETextDocumentContext;
OBJC_CLASS UIWKDocumentContext;

namespace WebKit {

struct DocumentEditingContextRequest {
    enum class Options : uint8_t {
        Text = 1 << 0,
        AttributedText = 1 << 1,
        Rects = 1 << 2,
        Spatial = 1 << 3,
        Annotation = 1 << 4,
        MarkedTextRects = 1 << 5,
        SpatialAndCurrentSelection = 1 << 6,
        AutocorrectedRanges = 1 << 7,
    };

    OptionSet<Options> options;

    WebCore::TextGranularity surroundingGranularity { WebCore::TextGranularity::CharacterGranularity };
    int64_t granularityCount { 0 };

    WebCore::FloatRect rect;

    std::optional<WebCore::ElementContext> textInputContext;
};

struct DocumentEditingContext {
    WKBETextDocumentContext *toPlatformContext(OptionSet<DocumentEditingContextRequest::Options>);
    UIWKDocumentContext *toLegacyPlatformContext(OptionSet<DocumentEditingContextRequest::Options>);

    WebCore::AttributedString contextBefore;
    WebCore::AttributedString selectedText;
    WebCore::AttributedString contextAfter;
    WebCore::AttributedString markedText;
    WebCore::AttributedString annotatedText;

    struct Range {
        uint64_t location { 0 };
        uint64_t length { 0 };
    };

    Range selectedRangeInMarkedText;

    struct TextRectAndRange {
        WebCore::FloatRect rect;
        Range range;
    };

    Vector<TextRectAndRange> textRects;
    Vector<Range> autocorrectedRanges;
};

}

#endif // PLATFORM(IOS_FAMILY)
