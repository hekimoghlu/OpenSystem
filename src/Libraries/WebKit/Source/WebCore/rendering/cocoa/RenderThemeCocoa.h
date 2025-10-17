/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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

#include "Icon.h"
#include "RenderTheme.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS NSDateComponentsFormatter;
struct AttachmentLayout;

namespace WebCore {

class RenderThemeCocoa : public RenderTheme {
public:
    WEBCORE_EXPORT static RenderThemeCocoa& singleton();

protected:
    virtual Color pictureFrameColor(const RenderObject&);
#if ENABLE(ATTACHMENT_ELEMENT)
    int attachmentBaseline(const RenderAttachment&) const final;
    void paintAttachmentText(GraphicsContext&, AttachmentLayout*) final;
#endif

    Color platformSpellingMarkerColor(OptionSet<StyleColorOptions>) const override;
    Color platformDictationAlternativesMarkerColor(OptionSet<StyleColorOptions>) const override;
    Color platformGrammarMarkerColor(OptionSet<StyleColorOptions>) const override;

private:
    void purgeCaches() override;

    bool shouldHaveCapsLockIndicator(const HTMLInputElement&) const final;

    void paintFileUploadIconDecorations(const RenderObject& inputRenderer, const RenderObject& buttonRenderer, const PaintInfo&, const IntRect&, Icon*, FileUploadDecorations) override;

    Seconds animationRepeatIntervalForProgressBar(const RenderProgress&) const final;

#if ENABLE(APPLE_PAY)
    void adjustApplePayButtonStyle(RenderStyle&, const Element*) const override;
#endif

#if ENABLE(VIDEO)
    String mediaControlsStyleSheet() override;
    Vector<String, 2> mediaControlsScripts() override;
    String mediaControlsBase64StringForIconNameAndType(const String&, const String&) override;
    String mediaControlsFormattedStringForDuration(double) override;

    String m_mediaControlsLocalizedStringsScript;
    String m_mediaControlsScript;
    String m_mediaControlsStyleSheet;
    RetainPtr<NSDateComponentsFormatter> m_durationFormatter;
#endif // ENABLE(VIDEO)
};

}
