/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#include "config.h"
#include "IntelligenceTextEffectsSupport.h"

#include "CharacterRange.h"
#include "DocumentInlines.h"
#include "DocumentMarkerController.h"
#include "FloatRect.h"
#include "RenderedDocumentMarker.h"
#include "SimpleRange.h"
#include "TextIndicator.h"
#include "TextIterator.h"
#include <wtf/UUID.h>

namespace WebCore {
namespace IntelligenceTextEffectsSupport {

#if ENABLE(WRITING_TOOLS)
Vector<FloatRect> writingToolsTextSuggestionRectsInRootViewCoordinates(Document& document, const SimpleRange& scope, const CharacterRange& range)
{
    auto resolvedRange = resolveCharacterRange(scope, range);

    Vector<FloatRect> textRectsInRootViewCoordinates;

    auto& markers = document.markers();
    markers.forEach(resolvedRange, { DocumentMarkerType::WritingToolsTextSuggestion }, [&](auto& node, auto& marker) {
        auto data = std::get<DocumentMarker::WritingToolsTextSuggestionData>(marker.data());

        auto markerRange = makeSimpleRange(node, marker);

        auto rect = document.view()->contentsToRootView(unionRect(RenderObject::absoluteTextRects(markerRange, { })));
        textRectsInRootViewCoordinates.append(WTFMove(rect));

        return false;
    });

    return textRectsInRootViewCoordinates;
}
#endif

void updateTextVisibility(Document& document, const SimpleRange& scope, const CharacterRange& range, bool visible, const WTF::UUID& identifier)
{
    if (visible) {
        document.markers().removeMarkers({ WebCore::DocumentMarkerType::TransparentContent }, [identifier](auto& marker) {
            auto& data = std::get<WebCore::DocumentMarker::TransparentContentData>(marker.data());
            return data.uuid == identifier ? WebCore::FilterMarkerResult::Remove : WebCore::FilterMarkerResult::Keep;
        });
    } else {
        auto resolvedRange = resolveCharacterRange(scope, range);
        document.markers().addTransparentContentMarker(resolvedRange, identifier);
    }
}

std::optional<TextIndicatorData> textPreviewDataForRange(Document&, const SimpleRange& scope, const CharacterRange& range)
{
    auto resolvedRange = resolveCharacterRange(scope, range);

    static constexpr OptionSet textIndicatorOptions {
        TextIndicatorOption::IncludeSnapshotOfAllVisibleContentWithoutSelection,
        TextIndicatorOption::ExpandClipBeyondVisibleRect,
        TextIndicatorOption::SkipReplacedContent,
        TextIndicatorOption::RespectTextColor,
        TextIndicatorOption::DoNotClipToVisibleRect,
#if PLATFORM(VISION)
        TextIndicatorOption::SnapshotContentAt3xBaseScale,
#endif
    };

    RefPtr textIndicator = WebCore::TextIndicator::createWithRange(resolvedRange, textIndicatorOptions, WebCore::TextIndicatorPresentationTransition::None, { });
    if (!textIndicator)
        return std::nullopt;

    return textIndicator->data();
}

#if ENABLE(WRITING_TOOLS)
void decorateWritingToolsTextReplacements(Document& document, const SimpleRange& scope, const CharacterRange& range)
{
    auto resolvedRange = resolveCharacterRange(scope, range);

    auto& markers = document.markers();

    Vector<std::tuple<SimpleRange, DocumentMarker::WritingToolsTextSuggestionData>> markersToReinsert;

    markers.forEach(resolvedRange, { DocumentMarkerType::WritingToolsTextSuggestion }, [&](auto& node, auto& marker) {
        auto range = makeSimpleRange(node, marker);
        auto data = std::get<DocumentMarker::WritingToolsTextSuggestionData>(marker.data());

        markersToReinsert.append({ range, data });

        return false;
    });

    markers.removeMarkers(resolvedRange, { DocumentMarkerType::WritingToolsTextSuggestion });

    for (const auto& [range, oldData] : markersToReinsert) {
        auto newData = DocumentMarker::WritingToolsTextSuggestionData { oldData.originalText, oldData.suggestionID, oldData.state, DocumentMarker::WritingToolsTextSuggestionData::Decoration::Underline };
        markers.addMarker(range, DocumentMarkerType::WritingToolsTextSuggestion, newData);
    }
}
#endif

} // namespace IntelligenceTextEffectsSupport
} // namespace WebCore
