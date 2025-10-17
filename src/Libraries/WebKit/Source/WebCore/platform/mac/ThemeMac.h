/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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

#include "ThemeCocoa.h"

#if PLATFORM(MAC)

namespace WebCore {

class ThemeMac final : public ThemeCocoa {
public:
    static bool supportsLargeFormControls();
    static void inflateControlPaintRect(StyleAppearance, FloatRect&, float, bool);

private:
    friend NeverDestroyed<ThemeMac>;
    ThemeMac() = default;

    std::optional<FontCascadeDescription> controlFont(StyleAppearance, const FontCascade&, float zoomFactor) const final;

    LengthSize controlSize(StyleAppearance, const FontCascade&, const LengthSize&, float zoomFactor) const final;
    LengthSize minimumControlSize(StyleAppearance, const FontCascade&, const LengthSize&, float zoomFactor) const final;

    LengthBox controlPadding(StyleAppearance, const FontCascade&, const LengthBox& zoomedBox, float zoomFactor) const final;
    LengthBox controlBorder(StyleAppearance, const FontCascade&, const LengthBox& zoomedBox, float zoomFactor) const final;

    bool controlRequiresPreWhiteSpace(StyleAppearance appearance) const final { return appearance == StyleAppearance::PushButton; }

    bool userPrefersContrast() const final;
    bool userPrefersReducedMotion() const final;
};

} // namespace WebCore

#endif // PLATFORM(MAC)
