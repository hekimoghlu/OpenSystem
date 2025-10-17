/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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

#include "DocumentMarker.h"
#include "Timer.h"
#include <memory>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class Document;
class FontCascade;
class LayoutPoint;
class Node;
class RenderedDocumentMarker;
class Text;

struct SimpleRange;

enum class RemovePartiallyOverlappingMarker : bool { No, Yes };
enum class FilterMarkerResult : bool { Keep, Remove };

class DocumentMarkerController final : public CanMakeCheckedPtr<DocumentMarkerController> {
    WTF_MAKE_TZONE_ALLOCATED(DocumentMarkerController);
    WTF_MAKE_NONCOPYABLE(DocumentMarkerController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DocumentMarkerController);
public:
    enum class IterationDirection : bool {
        Forwards, Backwards
    };

    DocumentMarkerController(Document&);
    ~DocumentMarkerController();

    void detach();

    WEBCORE_EXPORT void addMarker(const SimpleRange&, DocumentMarkerType, const DocumentMarker::Data& = { });
    void addMarker(Node&, unsigned startOffset, unsigned length, DocumentMarkerType, DocumentMarker::Data&& = { });
    WEBCORE_EXPORT void addMarker(Node&, DocumentMarker&&);
    void addDraggedContentMarker(const SimpleRange&);
    WEBCORE_EXPORT void addTransparentContentMarker(const SimpleRange&, WTF::UUID);

    void copyMarkers(Node& source, OffsetRange, Node& destination);
    bool hasMarkers() const;
    WEBCORE_EXPORT bool hasMarkers(const SimpleRange&, OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers());

    // When a marker partially overlaps with range, if removePartiallyOverlappingMarkers is true, we completely
    // remove the marker. If the argument is false, we will adjust the span of the marker so that it retains
    // the portion that is outside of the range.
    WEBCORE_EXPORT void removeMarkers(const SimpleRange&, OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers(), RemovePartiallyOverlappingMarker = RemovePartiallyOverlappingMarker::No);
    WEBCORE_EXPORT void removeMarkers(Node&, OffsetRange, OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers(), const Function<FilterMarkerResult(const DocumentMarker&)>& filterFunction = nullptr, RemovePartiallyOverlappingMarker = RemovePartiallyOverlappingMarker::No);

    WEBCORE_EXPORT void filterMarkers(const SimpleRange&, const Function<FilterMarkerResult(const DocumentMarker&)>& filterFunction, OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers(), RemovePartiallyOverlappingMarker = RemovePartiallyOverlappingMarker::No);

    WEBCORE_EXPORT void removeMarkers(OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers());
    WEBCORE_EXPORT void removeMarkers(OptionSet<DocumentMarkerType>, const Function<FilterMarkerResult(const RenderedDocumentMarker&)>& filterFunction);
    void removeMarkers(Node&, OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers());
    void repaintMarkers(OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers());
    void shiftMarkers(Node&, unsigned startOffset, int delta);

    void dismissMarkers(OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers());

    WEBCORE_EXPORT Vector<WeakPtr<RenderedDocumentMarker>> markersFor(Node&, OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers()) const;
    WEBCORE_EXPORT Vector<WeakPtr<RenderedDocumentMarker>> markersInRange(const SimpleRange&, OptionSet<DocumentMarkerType>);
    WEBCORE_EXPORT Vector<SimpleRange> rangesForMarkersInRange(const SimpleRange&, OptionSet<DocumentMarkerType>);
    void clearDescriptionOnMarkersIntersectingRange(const SimpleRange&, OptionSet<DocumentMarkerType>);

    WEBCORE_EXPORT void updateRectsForInvalidatedMarkersOfType(DocumentMarkerType);

    void invalidateRectsForAllMarkers();
    void invalidateRectsForMarkersInNode(Node&);

    WeakPtr<DocumentMarker> markerContainingPoint(const LayoutPoint&, DocumentMarkerType);
    WEBCORE_EXPORT Vector<FloatRect> renderedRectsForMarkers(DocumentMarkerType);

    template<IterationDirection = IterationDirection::Forwards>
    WEBCORE_EXPORT void forEach(const SimpleRange&, OptionSet<DocumentMarkerType>, Function<bool(Node&, RenderedDocumentMarker&)>&&);

    WEBCORE_EXPORT static std::tuple<float, float> markerYPositionAndHeightForFont(const FontCascade&);

#if ENABLE(TREE_DEBUGGING)
    void showMarkers() const;
#endif

private:
    struct TextRange {
        Ref<Node> node;
        OffsetRange range;
    };
    Vector<TextRange> collectTextRanges(const SimpleRange&);

    using MarkerMap = UncheckedKeyHashMap<Ref<Node>, std::unique_ptr<Vector<RenderedDocumentMarker>>>;

    bool possiblyHasMarkers(OptionSet<DocumentMarkerType>) const;
    OptionSet<DocumentMarkerType> removeMarkersFromList(MarkerMap::iterator, OptionSet<DocumentMarkerType>, const Function<FilterMarkerResult(const RenderedDocumentMarker&)>& filterFunction = nullptr);

    void forEachOfTypes(OptionSet<DocumentMarkerType>, Function<bool(Node&, RenderedDocumentMarker&)>&&);

    void applyToCollapsedRangeMarker(const SimpleRange&, OptionSet<DocumentMarkerType>, Function<bool(Node&, RenderedDocumentMarker&)>&&);

    void fadeAnimationTimerFired();
    void writingToolsTextSuggestionAnimationTimerFired();

    Ref<Document> protectedDocument() const;

    MarkerMap m_markers;
    // Provide a quick way to determine whether a particular marker type is absent without going through the map.
    OptionSet<DocumentMarkerType> m_possiblyExistingMarkerTypes;
    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;

    Timer m_fadeAnimationTimer;
    Timer m_writingToolsTextSuggestionAnimationTimer;
};


template<>
WEBCORE_EXPORT void DocumentMarkerController::forEach<DocumentMarkerController::IterationDirection::Forwards>(const SimpleRange&, OptionSet<DocumentMarkerType>, Function<bool(Node&, RenderedDocumentMarker&)>&&);

template<>
WEBCORE_EXPORT void DocumentMarkerController::forEach<DocumentMarkerController::IterationDirection::Backwards>(const SimpleRange&, OptionSet<DocumentMarkerType>, Function<bool(Node&, RenderedDocumentMarker&)>&&);

WEBCORE_EXPORT void addMarker(const SimpleRange&, DocumentMarkerType, const DocumentMarker::Data& = { });
void addMarker(Node&, unsigned startOffset, unsigned length, DocumentMarkerType, DocumentMarker::Data&& = { });
void removeMarkers(const SimpleRange&, OptionSet<DocumentMarkerType> = DocumentMarker::allMarkers(), RemovePartiallyOverlappingMarker = RemovePartiallyOverlappingMarker::No);

WEBCORE_EXPORT SimpleRange makeSimpleRange(Node&, const DocumentMarker&);

inline bool DocumentMarkerController::hasMarkers() const
{
    ASSERT(m_markers.isEmpty() == !m_possiblyExistingMarkerTypes.containsAny(DocumentMarker::allMarkers()));
    return !m_markers.isEmpty();
}

} // namespace WebCore

#if ENABLE(TREE_DEBUGGING)
void showDocumentMarkers(const WebCore::DocumentMarkerController*);
#endif
