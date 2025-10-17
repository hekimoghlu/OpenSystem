/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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

namespace WTF {
class TextStream;
}

namespace WebCore {

// Must follow CSSValueKeywords.in order
enum class StyleAppearance : uint8_t {
    None,
    Auto,
    Base,
    Checkbox,
    Radio,
    PushButton,
    SquareButton,
    Button,
    DefaultButton,
    Listbox,
    Menulist,
    MenulistButton,
    Meter,
    ProgressBar,
    SliderHorizontal,
    SliderVertical,
    SearchField,
#if ENABLE(APPLE_PAY)
    ApplePayButton,
#endif
#if ENABLE(ATTACHMENT_ELEMENT)
    Attachment,
    BorderlessAttachment,
#endif
    TextArea,
    TextField,
    // Internal-only Values
    ColorWell,
#if ENABLE(SERVICE_CONTROLS)
    ImageControlsButton,
#endif
    InnerSpinButton,
    ListButton,
    SearchFieldDecoration,
    SearchFieldResultsDecoration,
    SearchFieldResultsButton,
    SearchFieldCancelButton,
    SliderThumbHorizontal,
    SliderThumbVertical,
    Switch,
    SwitchThumb,
    SwitchTrack
};

WTF::TextStream& operator<<(WTF::TextStream&, StyleAppearance);

} // namespace WebCore
