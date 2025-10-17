/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
#include "AcceleratedBackingStore.h"

#include "HardwareAccelerationManager.h"
#include "WebPageProxy.h"
#include <WebCore/PlatformDisplay.h>
#include <gtk/gtk.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/glib/GUniquePtr.h>

#if PLATFORM(GTK)
#include "AcceleratedBackingStoreDMABuf.h"
#endif

namespace WebKit {
using namespace WebCore;

#if PLATFORM(GTK)
static bool gtkCanUseHardwareAcceleration()
{
    static bool canUseHardwareAcceleration;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        GUniqueOutPtr<GError> error;
#if USE(GTK4)
        canUseHardwareAcceleration = gdk_display_prepare_gl(gdk_display_get_default(), &error.outPtr());
#else
        auto* window = gtk_window_new(GTK_WINDOW_POPUP);
        gtk_widget_realize(window);
        auto context = adoptGRef(gdk_window_create_gl_context(gtk_widget_get_window(window), &error.outPtr()));
        canUseHardwareAcceleration = !!context;
        gtk_widget_destroy(window);
#endif
        if (!canUseHardwareAcceleration)
            g_warning("Disabled hardware acceleration because GTK failed to initialize GL: %s.", error->message);
    });
    return canUseHardwareAcceleration;
}
#endif

WTF_MAKE_TZONE_ALLOCATED_IMPL(AcceleratedBackingStore);

bool AcceleratedBackingStore::checkRequirements()
{
#if PLATFORM(GTK)
    if (AcceleratedBackingStoreDMABuf::checkRequirements())
        return gtkCanUseHardwareAcceleration();
#endif

    return false;
}

RefPtr<AcceleratedBackingStore> AcceleratedBackingStore::create(WebPageProxy& webPage)
{
    if (!HardwareAccelerationManager::singleton().canUseHardwareAcceleration())
        return nullptr;

#if PLATFORM(GTK)
    if (AcceleratedBackingStoreDMABuf::checkRequirements())
        return AcceleratedBackingStoreDMABuf::create(webPage);
#endif
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

AcceleratedBackingStore::AcceleratedBackingStore(WebPageProxy& webPage)
    : m_webPage(webPage)
{
}

} // namespace WebKit
