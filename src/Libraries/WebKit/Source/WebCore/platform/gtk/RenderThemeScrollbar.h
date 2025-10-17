/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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

#if !USE(GTK4) && USE(CAIRO)

#include "RenderThemeGadget.h"
#include <gtk/gtk.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderThemeScrollbar {
    WTF_MAKE_TZONE_ALLOCATED(RenderThemeScrollbar);
    WTF_MAKE_NONCOPYABLE(RenderThemeScrollbar);
public:
    enum class Type {
        VerticalScrollbarRight = 1,
        VerticalScrollbarLeft,
        HorizontalScrollbar,
        VerticalScrollIndicatorRight,
        VerticalScrollIndicatorLeft,
        HorizontalScrollIndicator,
    };
    static RenderThemeScrollbar& getOrCreate(Type);
    static void clearCache();

    enum class Mode { Full, Indicator };
    enum class VerticalPosition { Right, Left };
    RenderThemeScrollbar(GtkOrientation, Mode, VerticalPosition = VerticalPosition::Right);
    ~RenderThemeScrollbar() = default;

    RenderThemeGadget& scrollbar() const { return *m_scrollbar; }
    RenderThemeGadget& contents() const { return *m_contents; }
    RenderThemeGadget& slider() const { return *m_slider; }
    RenderThemeGadget& trough() const { return *m_contents->child(m_troughPosition); }
    RenderThemeGadget* stepper(RenderThemeScrollbarGadget::Steppers);

private:
    std::unique_ptr<RenderThemeGadget> m_scrollbar;
    std::unique_ptr<RenderThemeBoxGadget> m_contents;
    std::unique_ptr<RenderThemeGadget> m_slider;
    unsigned m_troughPosition;
    unsigned m_steppersPosition[4];
};

} // namespace WebCore

#endif // !USE(GTK4) && USE(CAIRO)
