/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

#include "Image.h"
#include "IntPoint.h"
#include <variant>
#include <wtf/Assertions.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

#if PLATFORM(WIN)
typedef struct HICON__* HICON;
typedef HICON HCURSOR;
#include <wtf/RefCounted.h>
#elif PLATFORM(COCOA)
#include <wtf/RetainPtr.h>
#elif PLATFORM(GTK)
#include "GRefPtrGtk.h"
#endif

#if HAVE(NSCURSOR)
OBJC_CLASS NSCursor;
#endif

#if PLATFORM(WIN)
typedef struct HICON__ *HICON;
typedef HICON HCURSOR;
#endif

namespace WebCore {

class Image;

#if PLATFORM(WIN)

class SharedCursor : public RefCounted<SharedCursor> {
public:
    static Ref<SharedCursor> create(HCURSOR);
    WEBCORE_EXPORT ~SharedCursor();
    HCURSOR nativeCursor() const { return m_nativeCursor; }

private:
    SharedCursor(HCURSOR);
    HCURSOR m_nativeCursor;
};

#endif

#if PLATFORM(WIN)
using PlatformCursor = RefPtr<SharedCursor>;
#elif HAVE(NSCURSOR)
using PlatformCursor = NSCursor *;
#elif PLATFORM(GTK)
using PlatformCursor = GRefPtr<GdkCursor>;
#else
using PlatformCursor = void*;
#endif

enum class PlatformCursorType : uint8_t {
    Invalid,
    Pointer,
    Cross,
    Hand,
    IBeam,
    Wait,
    Help,
    EastResize,
    NorthResize,
    NorthEastResize,
    NorthWestResize,
    SouthResize,
    SouthEastResize,
    SouthWestResize,
    WestResize,
    NorthSouthResize,
    EastWestResize,
    NorthEastSouthWestResize,
    NorthWestSouthEastResize,
    ColumnResize,
    RowResize,
    MiddlePanning,
    EastPanning,
    NorthPanning,
    NorthEastPanning,
    NorthWestPanning,
    SouthPanning,
    SouthEastPanning,
    SouthWestPanning,
    WestPanning,
    Move,
    VerticalText,
    Cell,
    ContextMenu,
    Alias,
    Progress,
    NoDrop,
    Copy,
    None,
    NotAllowed,
    ZoomIn,
    ZoomOut,
    Grab,
    Grabbing,
    Custom
};

class Cursor {
    WTF_MAKE_TZONE_ALLOCATED(Cursor);
public:
    using Type = PlatformCursorType;

    struct CustomCursorIPCData {
        Ref<Image> image;
        IntPoint hotSpot;
#if ENABLE(MOUSE_CURSOR_SCALE)
        float scaleFactor { 0 };
#endif
    };
    using IPCData = std::variant<Type /* Non custom type */, std::optional<CustomCursorIPCData>>;

    Cursor() = default;
    static std::optional<Cursor> fromIPCData(IPCData&&);

    WEBCORE_EXPORT static const Cursor& fromType(Cursor::Type);

    WEBCORE_EXPORT Cursor(Image*, const IntPoint& hotSpot);

#if ENABLE(MOUSE_CURSOR_SCALE)
    // Hot spot is in image pixels.
    WEBCORE_EXPORT Cursor(Image*, const IntPoint& hotSpot, float imageScaleFactor);
#endif

    explicit Cursor(Type);

    IPCData ipcData() const;

    Type type() const;
    RefPtr<Image> image() const { return m_image; }
    const IntPoint& hotSpot() const { return m_hotSpot; }

#if ENABLE(MOUSE_CURSOR_SCALE)
    // Image scale in image pixels per logical (UI) pixel.
    float imageScaleFactor() const { return m_imageScaleFactor; }
#endif

    WEBCORE_EXPORT PlatformCursor platformCursor() const;

    WEBCORE_EXPORT void setAsPlatformCursor() const;

private:
    void ensurePlatformCursor() const;

    Type m_type { Type::Invalid };
    RefPtr<Image> m_image;
    IntPoint m_hotSpot;

#if ENABLE(MOUSE_CURSOR_SCALE)
    float m_imageScaleFactor { 1 };
#endif

#if !HAVE(NSCURSOR)
    mutable PlatformCursor m_platformCursor { nullptr };
#else
    mutable RetainPtr<NSCursor> m_platformCursor;
#endif

};

IntPoint determineHotSpot(Image*, const IntPoint& specifiedHotSpot);

WEBCORE_EXPORT const Cursor& pointerCursor();
const Cursor& crossCursor();
WEBCORE_EXPORT const Cursor& handCursor();
const Cursor& moveCursor();
WEBCORE_EXPORT const Cursor& iBeamCursor();
const Cursor& waitCursor();
const Cursor& helpCursor();
const Cursor& eastResizeCursor();
const Cursor& northResizeCursor();
const Cursor& northEastResizeCursor();
const Cursor& northWestResizeCursor();
const Cursor& southResizeCursor();
const Cursor& southEastResizeCursor();
const Cursor& southWestResizeCursor();
const Cursor& westResizeCursor();
const Cursor& northSouthResizeCursor();
const Cursor& eastWestResizeCursor();
const Cursor& northEastSouthWestResizeCursor();
const Cursor& northWestSouthEastResizeCursor();
const Cursor& columnResizeCursor();
const Cursor& rowResizeCursor();
const Cursor& middlePanningCursor();
const Cursor& eastPanningCursor();
const Cursor& northPanningCursor();
const Cursor& northEastPanningCursor();
const Cursor& northWestPanningCursor();
const Cursor& southPanningCursor();
const Cursor& southEastPanningCursor();
const Cursor& southWestPanningCursor();
const Cursor& westPanningCursor();
const Cursor& verticalTextCursor();
const Cursor& cellCursor();
const Cursor& contextMenuCursor();
const Cursor& noDropCursor();
const Cursor& notAllowedCursor();
const Cursor& progressCursor();
const Cursor& aliasCursor();
const Cursor& zoomInCursor();
const Cursor& zoomOutCursor();
const Cursor& copyCursor();
const Cursor& noneCursor();
const Cursor& grabCursor();
const Cursor& grabbingCursor();

inline Cursor::Type Cursor::type() const
{
    ASSERT(m_type > Type::Invalid);
    ASSERT(m_type <= Type::Custom);
    return m_type;
}

inline std::optional<Cursor> Cursor::fromIPCData(IPCData&& ipcData)
{
    return WTF::switchOn(WTFMove(ipcData), [](Type&& type) -> std::optional<Cursor> {
        if (type == Type::Invalid || type == Type::Custom)
            return std::nullopt;
        auto& cursorReference = Cursor::fromType(type);
        // Calling platformCursor here will eagerly create the platform cursor for the cursor singletons inside WebCore.
        // This will avoid having to re-create the platform cursors over and over.
        (void)cursorReference.platformCursor();
        return cursorReference;
    }, [](std::optional<CustomCursorIPCData>&& imageData) -> std::optional<Cursor> {
        if (!imageData)
            return Cursor { &Image::nullImage(), IntPoint() };
        ASSERT(imageData->image->rect().contains(imageData->hotSpot));
#if ENABLE(MOUSE_CURSOR_SCALE)
        return Cursor(imageData->image.ptr(), imageData->hotSpot, imageData->scaleFactor);
#else
        return Cursor(imageData->image.ptr(), imageData->hotSpot);
#endif
    });
}

inline auto Cursor::ipcData() const -> IPCData
{
    auto type = this->type();
    if (type != Type::Custom)
        return type;
    if (m_image->isNull())
        return std::nullopt;
    return CustomCursorIPCData {
        *m_image
        , m_hotSpot
#if ENABLE(MOUSE_CURSOR_SCALE)
        , m_imageScaleFactor
#endif
    };
}

} // namespace WebCore
