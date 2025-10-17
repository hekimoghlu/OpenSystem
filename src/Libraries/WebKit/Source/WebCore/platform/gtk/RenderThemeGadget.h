/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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

#include "Color.h"
#include "IntSize.h"
#include <gtk/gtk.h>
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/CString.h>

namespace WebCore {
class FloatRect;

class RenderThemeGadget {
    WTF_MAKE_TZONE_ALLOCATED(RenderThemeGadget);
    WTF_MAKE_NONCOPYABLE(RenderThemeGadget);
public:
    enum class Type {
        Generic,
        Scrollbar,
    };

    struct Info {
        Type type;
        const char* name;
        Vector<const char*> classList;
    };

    static std::unique_ptr<RenderThemeGadget> create(const Info&, RenderThemeGadget* parent = nullptr, const Vector<RenderThemeGadget::Info> siblings = Vector<RenderThemeGadget::Info>(), unsigned position = 0);
    RenderThemeGadget(const Info&, RenderThemeGadget* parent, const Vector<RenderThemeGadget::Info> siblings, unsigned position);
    virtual ~RenderThemeGadget();

    virtual IntSize preferredSize() const;
    virtual bool render(cairo_t*, const FloatRect&, FloatRect* = nullptr);
    virtual IntSize minimumSize() const;

    GtkBorder contentsBox() const;
    Color color() const;
    Color backgroundColor() const;
    double opacity() const;

    GtkStyleContext* context() const { return m_context.get(); }

    GtkStateFlags state() const;
    void setState(GtkStateFlags);

protected:
    GtkBorder marginBox() const;
    GtkBorder borderBox() const;
    GtkBorder paddingBox() const;

    GRefPtr<GtkStyleContext> m_context;
};

class RenderThemeBoxGadget final : public RenderThemeGadget {
public:
    RenderThemeBoxGadget(const RenderThemeGadget::Info&, GtkOrientation, const Vector<RenderThemeGadget::Info> children, RenderThemeGadget* parent = nullptr);

    IntSize preferredSize() const override;

    RenderThemeGadget* child(unsigned index) const { return m_children[index].get(); }

private:
    Vector<std::unique_ptr<RenderThemeGadget>> m_children;
    GtkOrientation m_orientation { GTK_ORIENTATION_HORIZONTAL };
};

class RenderThemeScrollbarGadget final : public RenderThemeGadget {
public:
    RenderThemeScrollbarGadget(const Info&, RenderThemeGadget* parent, const Vector<RenderThemeGadget::Info> siblings, unsigned position);

    enum class Steppers {
        Backward = 1 << 0,
        Forward = 1 << 1,
        SecondaryBackward = 1 << 2,
        SecondaryForward = 1 << 3
    };
    OptionSet<Steppers> steppers() const { return m_steppers; };

    void renderStepper(cairo_t*, const FloatRect&, RenderThemeGadget*, GtkOrientation, Steppers);

private:
    OptionSet<Steppers> m_steppers;
};

} // namespace WebCore

#endif // !USE(GTK4) && USE(CAIRO)
