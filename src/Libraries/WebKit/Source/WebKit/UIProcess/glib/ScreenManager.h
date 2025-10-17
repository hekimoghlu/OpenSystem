/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 4, 2022.
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

#if PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))

#include <WebCore/ScreenProperties.h>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Vector.h>
#include <wtf/glib/GRefPtr.h>

#if PLATFORM(GTK)
typedef struct _GdkMonitor GdkMonitor;
using PlatformScreen = GdkMonitor;
#elif PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
typedef struct _WPEScreen WPEScreen;
using PlatformScreen = WPEScreen;
#endif

namespace WebKit {

using PlatformDisplayID = uint32_t;

class ScreenManager {
    WTF_MAKE_NONCOPYABLE(ScreenManager);
    friend NeverDestroyed<ScreenManager>;
public:
    static ScreenManager& singleton();

    PlatformDisplayID displayID(PlatformScreen*) const;
    PlatformScreen* screen(PlatformDisplayID) const;
    PlatformDisplayID primaryDisplayID() const { return m_primaryDisplayID; }

    WebCore::ScreenProperties collectScreenProperties() const;

private:
    ScreenManager();

    static PlatformDisplayID generatePlatformDisplayID(PlatformScreen*);

    void addScreen(PlatformScreen*);
    void removeScreen(PlatformScreen*);
    void updatePrimaryDisplayID();
    void propertiesDidChange() const;

    Vector<GRefPtr<PlatformScreen>, 1> m_screens;
    HashMap<PlatformScreen*, PlatformDisplayID> m_screenToDisplayIDMap;
    PlatformDisplayID m_primaryDisplayID { 0 };
};

} // namespace WebKit

#endif // PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))
