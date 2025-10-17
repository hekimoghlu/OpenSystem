/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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

#if ENABLE(VIDEO)

#include "HTMLDivElement.h"
#include "MediaControllerInterface.h"
#include "TextTrackRepresentation.h"
#include <wtf/LoggerHelper.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class HTMLMediaElement;
class VTTCue;

class MediaControlTextTrackContainerElement final
    : public HTMLDivElement
    , public TextTrackRepresentationClient
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaControlTextTrackContainerElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MediaControlTextTrackContainerElement);
public:
    static Ref<MediaControlTextTrackContainerElement> create(Document&, HTMLMediaElement&);

    enum class ForceUpdate : bool { No, Yes };
    void updateSizes(ForceUpdate force = ForceUpdate::No);
    void updateDisplay();

    TextTrackRepresentation* textTrackRepresentation() const { return m_textTrackRepresentation.get(); }
    void updateTextTrackRepresentationImageIfNeeded();
    void requiresTextTrackRepresentationChanged();

    void enteredFullscreen();
    void exitedFullscreen();

private:
    explicit MediaControlTextTrackContainerElement(Document&, HTMLMediaElement&);

    // Element
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;

    // TextTrackRepresentationClient
    RefPtr<NativeImage> createTextTrackRepresentationImage() override;
    void textTrackRepresentationBoundsChanged(const IntRect&) override;

    void updateTextTrackRepresentationIfNeeded();
    void clearTextTrackRepresentation();

    bool updateVideoDisplaySize();
    void updateActiveCuesFontSize();
    void updateTextStrokeStyle();
    void processActiveVTTCue(VTTCue&);
    void updateTextTrackStyle();

    void hide();
    void show();
    bool isShowing() const;

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final;
    uint64_t logIdentifier() const final;
    WTFLogChannel& logChannel() const final;
    ASCIILiteral logClassName() const final { return "MediaControlTextTrackContainerElement"_s; }
    mutable RefPtr<Logger> m_logger;
    mutable uint64_t m_logIdentifier { 0 };
#endif

    std::unique_ptr<TextTrackRepresentation> m_textTrackRepresentation;

    WeakPtr<HTMLMediaElement> m_mediaElement;
    IntRect m_videoDisplaySize;
    int m_fontSize { 0 };
    bool m_fontSizeIsImportant { false };
    bool m_needsToGenerateTextTrackRepresentation { false };
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
