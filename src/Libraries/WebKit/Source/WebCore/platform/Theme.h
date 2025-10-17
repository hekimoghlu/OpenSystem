/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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

#include "ControlStyle.h"
#include "ThemeTypes.h"
#include <optional>
#include <wtf/Forward.h>

namespace WebCore {

class FloatRect;
class FloatSize;
class FontCascade;
class FontCascadeDescription;
class GraphicsContext;
class ScrollView;

struct LengthSize;

class Theme {
public:
    static Theme& singleton();

    // The font description result should have a zoomed font size.
    virtual std::optional<FontCascadeDescription> controlFont(StyleAppearance, const FontCascade&, float) const;

    // The size here is in zoomed coordinates already. If a new size is returned, it also needs to be in zoomed coordinates.
    virtual LengthSize controlSize(StyleAppearance, const FontCascade&, const LengthSize&, float) const;

    // Returns the minimum size for a control in zoomed coordinates.
    LengthSize minimumControlSize(StyleAppearance, const FontCascade&, const LengthSize& zoomedSize, const LengthSize& nonShrinkableZoomedSize, float zoomFactor) const;
    
    // Allows the theme to modify the existing padding/border.
    virtual LengthBox controlPadding(StyleAppearance, const FontCascade&, const LengthBox& zoomedBox, float zoomFactor) const;
    virtual LengthBox controlBorder(StyleAppearance, const FontCascade&, const LengthBox& zoomedBox, float zoomFactor) const;

    // Whether or not whitespace: pre should be forced on always.
    virtual bool controlRequiresPreWhiteSpace(StyleAppearance) const { return false; }

    virtual void drawNamedImage(const String&, GraphicsContext&, const FloatSize&) const;

    virtual bool userPrefersContrast() const { return false; }
    virtual bool userPrefersReducedMotion() const { return false; }
    virtual bool userPrefersOnOffLabels() const { return false; }

protected:
    Theme() = default;
    virtual ~Theme() = default;

    virtual LengthSize minimumControlSize(StyleAppearance, const FontCascade&, const LengthSize& zoomedSize, float zoomFactor) const;

private:
    Theme(const Theme&) = delete;
    void operator=(const Theme&) = delete;
};

} // namespace WebCore
