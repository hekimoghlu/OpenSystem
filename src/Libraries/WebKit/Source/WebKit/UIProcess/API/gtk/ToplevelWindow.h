/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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

#include <gtk/gtk.h>
#include <wtf/HashSet.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

typedef struct _WebKitWebViewBase WebKitWebViewBase;

namespace WebKit {

class ToplevelWindow {
    WTF_MAKE_TZONE_ALLOCATED(ToplevelWindow);
    WTF_MAKE_NONCOPYABLE(ToplevelWindow);
public:
    static ToplevelWindow* forGtkWindow(GtkWindow*);
    explicit ToplevelWindow(GtkWindow*);
    ~ToplevelWindow();

    void addWebView(WebKitWebViewBase*);
    void removeWebView(WebKitWebViewBase*);

    GtkWindow* window() const { return m_window; }
    bool isActive() const;
    bool isFullscreen() const;
    bool isMinimized() const;
    bool isSuspended() const;

    GdkMonitor* monitor() const;
    bool isInMonitor() const;

private:
    void connectSignals();
    void disconnectSignals();
#if USE(GTK4)
    void connectSurfaceSignals();
    void disconnectSurfaceSignals();
#endif

    void notifyIsActive(bool);
    void notifyState(uint32_t, uint32_t);
    void notifyMonitorChanged(GdkMonitor*);

    GtkWindow* m_window { nullptr };
    HashSet<WebKitWebViewBase*> m_webViews;
#if USE(GTK4)
    GdkToplevelState m_state { static_cast<GdkToplevelState>(0) };
    HashSet<GdkMonitor*> m_monitors;
#endif
};

} // namespace WebKit
