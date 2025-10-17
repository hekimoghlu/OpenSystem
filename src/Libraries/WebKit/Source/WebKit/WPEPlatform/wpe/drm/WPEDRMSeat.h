/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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

#include "WPEKeymap.h"
#include "WPEView.h"
#include <libinput.h>
#include <wtf/HashMap.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GWeakPtr.h>

struct udev;

namespace WPE {

namespace DRM {

class Session;

class Seat {
    WTF_MAKE_TZONE_ALLOCATED(Seat);
public:
    static std::unique_ptr<Seat> create(struct udev*, Session&);
    explicit Seat(struct libinput*);
    ~Seat();

    void setView(WPEView* view);

private:
    WPEModifiers modifiers() const;
    void processEvents();
    void processEvent(struct libinput_event*);
    void handlePointerMotionEvent(struct libinput_event_pointer*);
    void handlePointerButtonEvent(struct libinput_event_pointer*);
    void handlePointerScrollWheelEvent(struct libinput_event_pointer*);
    void handlePointerScrollContinuousEvent(struct libinput_event_pointer*, WPEInputSource);
    void handleKeyEvent(struct libinput_event_keyboard*);
    void handleKey(uint32_t time, uint32_t key, bool pressed, bool fromRepeat);
    void handleTouchDownEvent(struct libinput_event_touch*);
    void handleTouchUpEvent(struct libinput_event_touch*);
    void handleTouchMotionEvent(struct libinput_event_touch*);
    void handleTouchCancelEvent(struct libinput_event_touch*);

    struct libinput* m_libinput { nullptr };
    GRefPtr<GSource> m_inputSource;
    GRefPtr<WPEKeymap> m_keymap;
    GWeakPtr<WPEView> m_view;

    struct {
        WPEInputSource source { WPE_INPUT_SOURCE_MOUSE };
        double x { 0 };
        double y { 0 };
        uint32_t modifiers { 0 };
        uint32_t time { 0 };
    } m_pointer;

    struct {
        WPEInputSource source { WPE_INPUT_SOURCE_KEYBOARD };
        uint32_t modifiers { 0 };
        uint32_t time { 0 };

        struct {
	    uint32_t key { 0 };
            GRefPtr<GSource> source;
            Seconds deadline;
        } repeat;
    } m_keyboard;

    struct {
        WPEInputSource source { WPE_INPUT_SOURCE_TOUCHSCREEN };
        uint32_t time { 0 };
        HashMap<int32_t, std::pair<double, double>, IntHash<int32_t>, WTF::SignedWithZeroKeyHashTraits<int32_t>> points;
    } m_touch;
};

} // namespace DRM

} // namespace WPE
