/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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

#include "WPEEvent.h"
#include "WPEKeymap.h"
#include "WPEToplevelWayland.h"
#include <wayland-client.h>
#include <wtf/HashMap.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GWeakPtr.h>

namespace WPE {

class WaylandSeat {
    WTF_MAKE_TZONE_ALLOCATED(WaylandSeat);
public:
    explicit WaylandSeat(struct wl_seat*);
    ~WaylandSeat();

    struct wl_seat* seat() const { return m_seat; }
    WPEKeymap* keymap() const { return m_keymap.get(); }
    uint32_t pointerModifiers() const { return m_pointer.modifiers; }
    std::pair<double, double> pointerCoords() const { return std::pair<double, double>(m_pointer.x, m_pointer.y); }

    void startListening();

    void setCursor(struct wl_surface*, int32_t, int32_t);

    void emitPointerEnter(WPEView*) const;
    void emitPointerLeave(WPEView*) const;

private:
    static const struct wl_seat_listener s_listener;
    static const struct wl_pointer_listener s_pointerListener;
    static const struct wl_keyboard_listener s_keyboardListener;
    static const struct wl_touch_listener s_touchListener;

    void updateCursor();
    WPEModifiers modifiers() const;
    void flushScrollEvent();
    void handleKeyEvent(uint32_t time, uint32_t key, uint32_t state, bool fromRepeat);
    bool keyRepeat(Seconds& delay, Seconds& interval);

    struct wl_seat* m_seat { nullptr };
    GRefPtr<WPEKeymap> m_keymap;
    struct {
        struct wl_pointer* object { nullptr };
        WPEInputSource source { WPE_INPUT_SOURCE_MOUSE };
        GWeakPtr<WPEToplevelWayland> toplevel;
        double x { 0 };
        double y { 0 };
        uint32_t modifiers { 0 };
        uint32_t time { 0 };
        uint32_t enterSerial { 0 };

        struct {
            WPEEvent* event { nullptr };
            double deltaX { 0 };
            double deltaY { 0 };
            int32_t valueX { 0 };
            int32_t valueY { 0 };
            bool isStop { false };
            WPEInputSource source { WPE_INPUT_SOURCE_MOUSE };

        } frame;
    } m_pointer;
    struct {
        struct wl_keyboard* object { nullptr };
        WPEInputSource source { WPE_INPUT_SOURCE_KEYBOARD };
        GWeakPtr<WPEToplevelWayland> toplevel;
        uint32_t modifiers { 0 };
        uint32_t time { 0 };

        struct {
            std::optional<int32_t> rate;
            std::optional<int32_t> delay;

            uint32_t key { 0 };
            GRefPtr<GSource> source;
            Seconds deadline;
        } repeat;

        struct {
            uint32_t key { 0 };
            unsigned keyval { 0 };
            uint32_t modifiers { 0 };
            uint32_t time { 0 };
        } capsLockUpEvent;
    } m_keyboard;
    struct {
        struct wl_touch* object { nullptr };
        WPEInputSource source { WPE_INPUT_SOURCE_TOUCHSCREEN };
        GWeakPtr<WPEToplevelWayland> toplevel;
        HashMap<int32_t, std::pair<double, double>, IntHash<int32_t>, WTF::SignedWithZeroKeyHashTraits<int32_t>> points;
    } m_touch;
};

} // namespace WPE
