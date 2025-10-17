/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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

#if ENABLE(WRITING_TOOLS)

#include <WebCore/CharacterRange.h>
#include <WebCore/SimpleRange.h>
#include <WebCore/TextIndicator.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UUID.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Range;

enum class TextIndicatorOption : uint16_t;
enum class TextAnimationRunMode : uint8_t;

}

namespace WebKit {

class WebPage;

struct TextAnimationRange {
    WTF::UUID animationUUID;
    WebCore::CharacterRange range;
};

struct TextAnimationUnanimatedRangeData {
    WTF::UUID animationUUID;
    WebCore::SimpleRange range;
};

struct ReplacedRangeAndString {
    WebCore::CharacterRange range;
    String string;
};

class TextAnimationController final {
    WTF_MAKE_TZONE_ALLOCATED(TextAnimationController);
    WTF_MAKE_NONCOPYABLE(TextAnimationController);

public:
    explicit TextAnimationController(WebPage&);

    void removeInitialTextAnimationForActiveWritingToolsSession();
    void addInitialTextAnimationForActiveWritingToolsSession();
    void addSourceTextAnimationForActiveWritingToolsSession(const WTF::UUID& sourceAnimationUUID, const WTF::UUID& destinationAnimationUUID, bool finished, const WebCore::CharacterRange&, const String&, CompletionHandler<void(WebCore::TextAnimationRunMode)>&&);
    void addDestinationTextAnimationForActiveWritingToolsSession(const WTF::UUID& sourceAnimationUUID, const WTF::UUID& destinationAnimationUUID, const std::optional<WebCore::CharacterRange>&, const String&);

    void saveSnapshotOfTextPlaceholderForAnimation(const WebCore::SimpleRange&);

    void clearAnimationsForActiveWritingToolsSession();

    void updateUnderlyingTextVisibilityForTextAnimationID(const WTF::UUID&, bool visible, CompletionHandler<void()>&& = [] { });

    std::optional<WebCore::TextIndicatorData> createTextIndicatorForRange(const WebCore::SimpleRange&);
    void createTextIndicatorForTextAnimationID(const WTF::UUID&, CompletionHandler<void(std::optional<WebCore::TextIndicatorData>&&)>&&);

private:
    std::optional<WebCore::SimpleRange> contextRangeForTextAnimationID(const WTF::UUID&) const;
    std::optional<WebCore::SimpleRange> contextRangeForActiveWritingToolsSession() const;
    std::optional<WebCore::SimpleRange> unreplacedRangeForActiveWritingToolsSession() const;

    void removeTransparentMarkersForTextAnimationID(const WTF::UUID&);
    void removeTransparentMarkersForActiveWritingToolsSession();

    RefPtr<WebCore::Document> document() const;
    WeakPtr<WebPage> m_webPage;

    std::optional<WTF::UUID> m_initialAnimationID;
    std::optional<TextAnimationUnanimatedRangeData> m_unanimatedRangeData;
    std::optional<ReplacedRangeAndString> m_alreadyReplacedRange;
    Vector<TextAnimationRange> m_textAnimationRanges;
    std::optional<WTF::UUID> m_activeAnimation;
    std::optional<CompletionHandler<void(WebCore::TextAnimationRunMode)>> m_finalReplaceHandler;
    std::optional<WebCore::TextIndicatorData> m_placeholderTextIndicatorData;

};

} // namespace WebKit

#endif // ENABLE(WRITING_TOOLS)
