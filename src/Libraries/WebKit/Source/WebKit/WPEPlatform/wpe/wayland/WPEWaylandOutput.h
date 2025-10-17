/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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

#include <wayland-client.h>
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

typedef struct _WPEViewWayland WPEViewWayland;

namespace WPE {

class WaylandOutput {
    WTF_MAKE_TZONE_ALLOCATED(WaylandOutput);
public:
    explicit WaylandOutput(struct wl_output*);
    ~WaylandOutput();

    struct wl_output* output() const { return m_output; }
    double scale() const { return m_scale; }

    void addScaleObserver(WPEViewWayland*, Function<void(WPEViewWayland*)>&&);
    void removeScaleObserver(WPEViewWayland*);

private:
    static const struct wl_output_listener s_listener;

    struct wl_output* m_output { nullptr };
    double m_scale { 1. };
    HashMap<WPEViewWayland*, Function<void(WPEViewWayland*)>> m_scaleObservers;
};

} // namespace WPE
