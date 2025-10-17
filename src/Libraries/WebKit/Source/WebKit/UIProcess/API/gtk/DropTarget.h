/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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

#if ENABLE(DRAG_SUPPORT)

#include <WebCore/DragActions.h>
#include <WebCore/IntPoint.h>
#include <WebCore/SelectionData.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/glib/GRefPtr.h>

typedef struct _GtkWidget GtkWidget;

#if USE(GTK4)
typedef struct _GdkDrop GdkDrop;
using PlatformDropContext = GdkDrop;
#else
typedef struct _GdkDragContext GdkDragContext;
typedef struct _GtkSelectionData GtkSelectionData;
using PlatformDropContext = GdkDragContext;
#endif

namespace WebKit {

class DropTarget {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DropTarget);
    WTF_MAKE_NONCOPYABLE(DropTarget);
public:
    explicit DropTarget(GtkWidget*);
    ~DropTarget();

    void didPerformAction();

private:
    void accept(PlatformDropContext*, std::optional<WebCore::IntPoint> = std::nullopt, unsigned = 0);
    void enter(WebCore::IntPoint&&, unsigned = 0);
    void update(WebCore::IntPoint&&, unsigned = 0);
    void leave();
    void drop(WebCore::IntPoint&&, unsigned = 0);

#if USE(GTK4)
    void loadData(const char* mimeType, CompletionHandler<void(GRefPtr<GBytes>&&)>&&);
    void loadData(CompletionHandler<void(Vector<String>&&)>&&);
    void didLoadData();
#else
    void dataReceived(WebCore::IntPoint&&, GtkSelectionData*, unsigned, unsigned);
    void leaveTimerFired();
#endif

    GtkWidget* m_webView { nullptr };
#if USE(GTK4)
    GRefPtr<GdkDrop> m_drop;
#else
    GRefPtr<GdkDragContext> m_drop;
#endif
    std::optional<WebCore::IntPoint> m_position;
    unsigned m_dataRequestCount { 0 };
    std::optional<WebCore::SelectionData> m_selectionData;
    std::optional<WebCore::DragOperation> m_operation;
#if USE(GTK4)
    GRefPtr<GCancellable> m_cancellable;
    StringBuilder m_uriListBuilder;
#else
    RunLoop::Timer m_leaveTimer;
#endif
};

} // namespace WebKit

#endif // ENABLE(DRAG_SUPPORT)
