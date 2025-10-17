/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 1, 2025.
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
#include "StyleAppearance.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

TextStream& operator<<(TextStream& ts, StyleAppearance appearance)
{
    switch (appearance) {
    case StyleAppearance::None:
        ts << "none";
        break;
    case StyleAppearance::Auto:
        ts << "auto";
        break;
    case StyleAppearance::Base:
        ts << "base";
        break;
    case StyleAppearance::Checkbox:
        ts << "checkbox";
        break;
    case StyleAppearance::Radio:
        ts << "radio";
        break;
    case StyleAppearance::PushButton:
        ts << "push-button";
        break;
    case StyleAppearance::SquareButton:
        ts << "square-button";
        break;
    case StyleAppearance::Button:
        ts << "button";
        break;
    case StyleAppearance::DefaultButton:
        ts << "default-button";
        break;
    case StyleAppearance::Listbox:
        ts << "listbox";
        break;
    case StyleAppearance::Menulist:
        ts << "menulist";
        break;
    case StyleAppearance::MenulistButton:
        ts << "menulist-button";
        break;
    case StyleAppearance::Meter:
        ts << "meter";
        break;
    case StyleAppearance::ProgressBar:
        ts << "progress-bar";
        break;
    case StyleAppearance::SliderHorizontal:
        ts << "slider-horizontal";
        break;
    case StyleAppearance::SliderVertical:
        ts << "slider-vertical";
        break;
    case StyleAppearance::SearchField:
        ts << "searchfield";
        break;
#if ENABLE(APPLE_PAY)
    case StyleAppearance::ApplePayButton:
        ts << "apple-pay-button";
        break;
#endif
#if ENABLE(ATTACHMENT_ELEMENT)
    case StyleAppearance::Attachment:
        ts << "attachment";
        break;
    case StyleAppearance::BorderlessAttachment:
        ts << "borderless-attachment";
        break;
#endif
    case StyleAppearance::TextArea:
        ts << "textarea";
        break;
    case StyleAppearance::TextField:
        ts << "textfield";
        break;
    case StyleAppearance::ColorWell:
        ts << "color-well";
        break;
#if ENABLE(SERVICE_CONTROLS)
    case StyleAppearance::ImageControlsButton:
        ts << "image-controls-button";
        break;
#endif
    case StyleAppearance::InnerSpinButton:
        ts << "inner-spin-button";
        break;
    case StyleAppearance::ListButton:
        ts << "list-button";
        break;
    case StyleAppearance::SearchFieldDecoration:
        ts << "searchfield-decoration";
        break;
    case StyleAppearance::SearchFieldResultsDecoration:
        ts << "searchfield-results-decoration";
        break;
    case StyleAppearance::SearchFieldResultsButton:
        ts << "searchfield-results-button";
        break;
    case StyleAppearance::SearchFieldCancelButton:
        ts << "searchfield-cancel-button";
        break;
    case StyleAppearance::SliderThumbHorizontal:
        ts << "sliderthumb-horizontal";
        break;
    case StyleAppearance::SliderThumbVertical:
        ts << "sliderthumb-vertical";
        break;
    case StyleAppearance::Switch:
        ts << "switch";
        break;
    case StyleAppearance::SwitchThumb:
        ts << "switch-thumb";
        break;
    case StyleAppearance::SwitchTrack:
        ts << "switch-track";
        break;
    }
    return ts;
}

} // namespace WebCore
