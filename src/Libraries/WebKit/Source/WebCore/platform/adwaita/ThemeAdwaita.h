/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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

#if USE(THEME_ADWAITA)

#include "Color.h"
#include "StyleColor.h"
#include "Theme.h"

namespace WebCore {

class ThemeAdwaita : public Theme {
public:
    ThemeAdwaita();

    virtual void platformColorsDidChange() { };

    bool userPrefersContrast() const final;
    bool userPrefersReducedMotion() const final;

    void setAccentColor(const Color&);
    Color accentColor();
private:
    LengthSize controlSize(StyleAppearance, const FontCascade&, const LengthSize&, float) const final;
    LengthSize minimumControlSize(StyleAppearance, const FontCascade&, const LengthSize&, float) const final;
    LengthBox controlBorder(StyleAppearance, const FontCascade&, const LengthBox&, float) const final;

#if PLATFORM(GTK) || PLATFORM(WPE)
    void refreshSettings();
#endif

    Color m_accentColor { SRGBA<uint8_t> { 52, 132, 228 } };

    bool m_prefersReducedMotion { false };
#if !USE(GTK4)
    bool m_prefersContrast { false };
#endif
};

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
