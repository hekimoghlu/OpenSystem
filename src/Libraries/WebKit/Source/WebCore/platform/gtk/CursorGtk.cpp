/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
#include "Cursor.h"

#include "Image.h"
#include "IntPoint.h"
#include "NotImplemented.h"
#include <gdk/gdk.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

#if USE(GTK4)
static GRefPtr<GdkCursor> fallbackCursor()
{
    static NeverDestroyed<GRefPtr<GdkCursor>> cursor(adoptGRef(gdk_cursor_new_from_name("default", nullptr)));
    return cursor;
}
#endif // USE(GTK4)

static GRefPtr<GdkCursor> createNamedCursor(const char* name)
{
#if USE(GTK4)
    return adoptGRef(gdk_cursor_new_from_name(name, fallbackCursor().get()));
#else
    return adoptGRef(gdk_cursor_new_from_name(gdk_display_get_default(), name));
#endif // USE(GTK4)
}

static GRefPtr<GdkCursor> createCustomCursor(Image* image, const IntPoint& hotSpot)
{
#if USE(GTK4)
    auto texture = image->adapter().gdkTexture();
    if (!texture)
        return nullptr;

    IntPoint effectiveHotSpot = determineHotSpot(image, hotSpot);
    return adoptGRef(gdk_cursor_new_from_texture(texture.get(), effectiveHotSpot.x(), effectiveHotSpot.y(), fallbackCursor().get()));
#else
    auto pixbuf = image->adapter().gdkPixbuf();
    if (!pixbuf)
        return nullptr;

    IntPoint effectiveHotSpot = determineHotSpot(image, hotSpot);
    return adoptGRef(gdk_cursor_new_from_pixbuf(gdk_display_get_default(), pixbuf.get(), effectiveHotSpot.x(), effectiveHotSpot.y()));
#endif // USE(GTK4)
}

void Cursor::ensurePlatformCursor() const
{
    if (m_platformCursor || m_type == Type::Pointer)
        return;

    switch (m_type) {
    case Type::Pointer:
        // A null GdkCursor is the default cursor for the window.
        m_platformCursor = 0;
        break;
    case Type::Cross:
        m_platformCursor = createNamedCursor("crosshair");
        break;
    case Type::Hand:
        m_platformCursor = createNamedCursor("pointer");
        break;
    case Type::IBeam:
        m_platformCursor = createNamedCursor("text");
        break;
    case Type::Wait:
        m_platformCursor = createNamedCursor("wait");
        break;
    case Type::Help:
        m_platformCursor = createNamedCursor("help");
        break;
    case Type::Move:
    case Type::MiddlePanning:
        m_platformCursor = createNamedCursor("move");
        break;
    case Type::EastResize:
    case Type::EastPanning:
        m_platformCursor = createNamedCursor("e-resize");
        break;
    case Type::NorthResize:
    case Type::NorthPanning:
        m_platformCursor = createNamedCursor("n-resize");
        break;
    case Type::NorthEastResize:
    case Type::NorthEastPanning:
        m_platformCursor = createNamedCursor("ne-resize");
        break;
    case Type::NorthWestResize:
    case Type::NorthWestPanning:
        m_platformCursor = createNamedCursor("nw-resize");
        break;
    case Type::SouthResize:
    case Type::SouthPanning:
        m_platformCursor = createNamedCursor("s-resize");
        break;
    case Type::SouthEastResize:
    case Type::SouthEastPanning:
        m_platformCursor = createNamedCursor("se-resize");
        break;
    case Type::SouthWestResize:
    case Type::SouthWestPanning:
        m_platformCursor = createNamedCursor("sw-resize");
        break;
    case Type::WestResize:
    case Type::WestPanning:
        m_platformCursor = createNamedCursor("w-resize");
        break;
    case Type::NorthSouthResize:
        m_platformCursor = createNamedCursor("ns-resize");
        break;
    case Type::EastWestResize:
        m_platformCursor = createNamedCursor("ew-resize");
        break;
    case Type::NorthEastSouthWestResize:
        m_platformCursor = createNamedCursor("nesw-resize");
        break;
    case Type::NorthWestSouthEastResize:
        m_platformCursor = createNamedCursor("nwse-resize");
        break;
    case Type::ColumnResize:
        m_platformCursor = createNamedCursor("col-resize");
        break;
    case Type::RowResize:
        m_platformCursor = createNamedCursor("row-resize");
        break;
    case Type::VerticalText:
        m_platformCursor = createNamedCursor("vertical-text");
        break;
    case Type::Cell:
        m_platformCursor = createNamedCursor("cell");
        break;
    case Type::ContextMenu:
        m_platformCursor = createNamedCursor("context-menu");
        break;
    case Type::Alias:
        m_platformCursor = createNamedCursor("alias");
        break;
    case Type::Progress:
        m_platformCursor = createNamedCursor("progress");
        break;
    case Type::NoDrop:
        m_platformCursor = createNamedCursor("no-drop");
        break;
    case Type::NotAllowed:
        m_platformCursor = createNamedCursor("not-allowed");
        break;
    case Type::Copy:
        m_platformCursor = createNamedCursor("copy");
        break;
    case Type::None:
        m_platformCursor = createNamedCursor("none");
        break;
    case Type::ZoomIn:
        m_platformCursor = createNamedCursor("zoom-in");
        break;
    case Type::ZoomOut:
        m_platformCursor = createNamedCursor("zoom-out");
        break;
    case Type::Grab:
        m_platformCursor = createNamedCursor("grab");
        break;
    case Type::Grabbing:
        m_platformCursor = createNamedCursor("grabbing");
        break;
    case Type::Custom:
        m_platformCursor = createCustomCursor(m_image.get(), m_hotSpot);
        break;
    case Type::Invalid:
        ASSERT_NOT_REACHED();
    }
}

}
