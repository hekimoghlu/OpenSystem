/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#include "WPEWaylandOutput.h"
#include <wtf/TZoneMallocInlines.h>

namespace WPE {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WaylandOutput);

WaylandOutput::WaylandOutput(struct wl_output* output)
    : m_output(output)
{
    wl_output_add_listener(m_output, &s_listener, this);
}

WaylandOutput::~WaylandOutput()
{
    if (wl_output_get_version(m_output) >= WL_OUTPUT_RELEASE_SINCE_VERSION)
        wl_output_release(m_output);
    else
        wl_output_destroy(m_output);
}

const struct wl_output_listener WaylandOutput::s_listener = {
    // geometry
    [](void*, struct wl_output*, int32_t, int32_t, int32_t, int32_t, int32_t, const char*, const char*, int32_t)
    {
    },
    // mode
    [](void*, struct wl_output*, uint32_t, int32_t, int32_t, int32_t)
    {
    },
    // done
    [](void*, struct wl_output*)
    {
    },
    // scale
    [](void* data, struct wl_output*, int32_t factor)
    {
        auto& output = *static_cast<WaylandOutput*>(data);
        if (output.m_scale == factor)
            return;

        output.m_scale = factor;
        for (const auto& it : output.m_scaleObservers)
            it.value(it.key);

    },
#ifdef WL_OUTPUT_NAME_SINCE_VERSION
    // name
    [](void*, struct wl_output*, const char*)
    {
    },
#endif
#ifdef WL_OUTPUT_DESCRIPTION_SINCE_VERSION
    // description
    [](void*, struct wl_output*, const char*)
    {
    },
#endif
};

void WaylandOutput::addScaleObserver(WPEViewWayland* view, Function<void(WPEViewWayland*)>&& observer)
{
    m_scaleObservers.set(view, WTFMove(observer));
}

void WaylandOutput::removeScaleObserver(WPEViewWayland* view)
{
    m_scaleObservers.remove(view);
}

} // namespace WPE
