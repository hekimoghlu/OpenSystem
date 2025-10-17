/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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

#include "PointerLockManager.h"

#if PLATFORM(WAYLAND)

#include "relative-pointer-unstable-v1-client-protocol.h"
#include <wayland-client.h>
#include <wtf/TZoneMalloc.h>

struct zwp_locked_pointer_v1;
struct zwp_pointer_constraints_v1;

namespace WebKit {

class WebPageProxy;

class PointerLockManagerWayland final : public PointerLockManager {
    WTF_MAKE_TZONE_ALLOCATED(PointerLockManagerWayland);
    WTF_MAKE_NONCOPYABLE(PointerLockManagerWayland);
public:
    PointerLockManagerWayland(WebPageProxy&, const WebCore::FloatPoint&, const WebCore::FloatPoint&, WebMouseEventButton, unsigned short, OptionSet<WebEventModifier>);
    ~PointerLockManagerWayland();

private:
    static const struct wl_registry_listener s_registryListener;
    static const struct zwp_relative_pointer_v1_listener s_relativePointerListener;

    bool lock() override;
    bool unlock() override;

    struct wl_registry* m_registry { nullptr };
    struct zwp_pointer_constraints_v1* m_pointerConstraints { nullptr };
    struct zwp_locked_pointer_v1* m_lockedPointer { nullptr };
    struct zwp_relative_pointer_manager_v1* m_relativePointerManager { nullptr };
    struct zwp_relative_pointer_v1* m_relativePointer { nullptr };
};

} // namespace WebKit

#endif // PLATFORM(WAYLAND)
