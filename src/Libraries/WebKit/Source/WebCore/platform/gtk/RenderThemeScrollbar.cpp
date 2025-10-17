/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#include "RenderThemeScrollbar.h"

#if !USE(GTK4) && USE(CAIRO)

#include "GtkUtilities.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderThemeScrollbar);

static UncheckedKeyHashMap<unsigned, std::unique_ptr<RenderThemeScrollbar>>& widgetMap()
{
    static NeverDestroyed<UncheckedKeyHashMap<unsigned, std::unique_ptr<RenderThemeScrollbar>>> map;
    return map;
}

RenderThemeScrollbar& RenderThemeScrollbar::getOrCreate(Type widgetType)
{
    auto addResult = widgetMap().ensure(static_cast<unsigned>(widgetType), [widgetType]() -> std::unique_ptr<RenderThemeScrollbar> {
        switch (widgetType) {
        case RenderThemeScrollbar::Type::VerticalScrollbarRight:
            return makeUnique<RenderThemeScrollbar>(GTK_ORIENTATION_VERTICAL, RenderThemeScrollbar::Mode::Full);
        case RenderThemeScrollbar::Type::VerticalScrollbarLeft:
            return makeUnique<RenderThemeScrollbar>(GTK_ORIENTATION_VERTICAL, RenderThemeScrollbar::Mode::Full, RenderThemeScrollbar::VerticalPosition::Left);
        case RenderThemeScrollbar::Type::HorizontalScrollbar:
            return makeUnique<RenderThemeScrollbar>(GTK_ORIENTATION_HORIZONTAL, RenderThemeScrollbar::Mode::Full);
        case RenderThemeScrollbar::Type::VerticalScrollIndicatorRight:
            return makeUnique<RenderThemeScrollbar>(GTK_ORIENTATION_VERTICAL, RenderThemeScrollbar::Mode::Indicator);
        case RenderThemeScrollbar::Type::VerticalScrollIndicatorLeft:
            return makeUnique<RenderThemeScrollbar>(GTK_ORIENTATION_VERTICAL, RenderThemeScrollbar::Mode::Indicator, RenderThemeScrollbar::VerticalPosition::Left);
        case RenderThemeScrollbar::Type::HorizontalScrollIndicator:
            return makeUnique<RenderThemeScrollbar>(GTK_ORIENTATION_HORIZONTAL, RenderThemeScrollbar::Mode::Indicator);
        }
        ASSERT_NOT_REACHED();
        return nullptr;
    });
    return *addResult.iterator->value;
}

void RenderThemeScrollbar::clearCache()
{
    widgetMap().clear();
}

RenderThemeScrollbar::RenderThemeScrollbar(GtkOrientation orientation, Mode mode, VerticalPosition verticalPosition)
{
    RenderThemeGadget::Info info = { RenderThemeGadget::Type::Scrollbar, "scrollbar", { } };
    if (orientation == GTK_ORIENTATION_VERTICAL) {
        info.classList.append("vertical");
        info.classList.append(verticalPosition == VerticalPosition::Right ? "right" : "left");
    } else {
        info.classList.append("horizontal");
        info.classList.append("bottom");
    }
    if (shouldUseOverlayScrollbars())
        info.classList.append("overlay-indicator");
    if (mode == Mode::Full)
        info.classList.append("hovering");
    m_scrollbar = RenderThemeGadget::create(info);

    Vector<RenderThemeGadget::Info> children;
    auto steppers = static_cast<RenderThemeScrollbarGadget*>(m_scrollbar.get())->steppers();
    if (steppers.contains(RenderThemeScrollbarGadget::Steppers::Backward)) {
        m_steppersPosition[0] = children.size();
        children.append({ RenderThemeGadget::Type::Generic, "button", { "up" } });
    }
    if (steppers.contains(RenderThemeScrollbarGadget::Steppers::SecondaryForward)) {
        m_steppersPosition[1] = children.size();
        children.append({ RenderThemeGadget::Type::Generic, "button", { "down" } });
    }
    m_troughPosition = children.size();
    children.append({ RenderThemeGadget::Type::Generic, "trough", { } });
    if (steppers.contains(RenderThemeScrollbarGadget::Steppers::SecondaryBackward)) {
        m_steppersPosition[2] = children.size();
        children.append({ RenderThemeGadget::Type::Generic, "button", { "up" } });
    }
    if (steppers.contains(RenderThemeScrollbarGadget::Steppers::Forward)) {
        m_steppersPosition[3] = children.size();
        children.append({ RenderThemeGadget::Type::Generic, "button", { "down" } });
    }
    info.type = RenderThemeGadget::Type::Generic;
    info.name = "contents";
    info.classList.clear();
    m_contents = makeUnique<RenderThemeBoxGadget>(info, orientation, children, m_scrollbar.get());
    info.name = "slider";
    m_slider = RenderThemeGadget::create(info, m_contents->child(m_troughPosition));
}

RenderThemeGadget* RenderThemeScrollbar::stepper(RenderThemeScrollbarGadget::Steppers scrollbarStepper)
{
    if (!static_cast<RenderThemeScrollbarGadget*>(m_scrollbar.get())->steppers().contains(scrollbarStepper))
        return nullptr;

    switch (scrollbarStepper) {
    case RenderThemeScrollbarGadget::Steppers::Backward:
        return m_contents->child(m_steppersPosition[0]);
    case RenderThemeScrollbarGadget::Steppers::SecondaryForward:
        return m_contents->child(m_steppersPosition[1]);
    case RenderThemeScrollbarGadget::Steppers::SecondaryBackward:
        return m_contents->child(m_steppersPosition[2]);
    case RenderThemeScrollbarGadget::Steppers::Forward:
        return m_contents->child(m_steppersPosition[3]);
    default:
        break;
    }

    return nullptr;
}

} // namespace WebCore

#endif // !USE(GTK4) && USE(CAIRO)
