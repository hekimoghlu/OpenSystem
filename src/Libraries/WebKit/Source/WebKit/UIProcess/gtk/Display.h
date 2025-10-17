/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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

#include <optional>
#include <wtf/Noncopyable.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/text/WTFString.h>

typedef struct _GdkDisplay GdkDisplay;

namespace WebCore {
class GLDisplay;
}

namespace WebKit {

class Display {
    WTF_MAKE_NONCOPYABLE(Display);
    friend class LazyNeverDestroyed<Display>;
public:
    static Display& singleton();
    ~Display();

    WebCore::GLDisplay* glDisplay() const;
    bool glDisplayIsSharedWithGtk() const { return glDisplay() && !m_glDisplayOwned; }

    bool isX11() const;
    bool isWayland() const;

    String accessibilityBusAddress() const;

private:
    Display();
#if PLATFORM(X11)
    bool initializeGLDisplayX11() const;
    String accessibilityBusAddressX11() const;
#endif
#if PLATFORM(WAYLAND)
    bool initializeGLDisplayWayland() const;
#endif

    GRefPtr<GdkDisplay> m_gdkDisplay;
    mutable std::unique_ptr<WebCore::GLDisplay> m_glDisplay;
    mutable bool m_glInitialized { false };
    mutable bool m_glDisplayOwned { false };
};

} // namespace WebKit
