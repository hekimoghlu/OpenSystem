/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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

#if USE(GLIB)

namespace WTF {

#if PLATFORM(GTK)

// This is a global enum to define priorities used by GLib run loop sources.
// In GLib, priorities are represented by an integer where lower values mean
// higher priority. The following macros are defined in GLib:
// G_PRIORITY_LOW = 300
// G_PRIORITY_DEFAULT_IDLE = 200
// G_PRIORITY_HIGH_IDLE = 100
// G_PRIORITY_DEFAULT = 0
// G_PRIORITY_HIGH = -100
// We don't use those macros here to avoid having to include glib header only
// for this. But we should take into account that GLib uses G_PRIORITY_DEFAULT
// for timeout sources and G_PRIORITY_DEFAULT_IDLE for idle sources.
// Changes in these priorities can have a huge impact in performance, and in
// the correctness too, so be careful when changing them.
enum RunLoopSourcePriority {
    // RunLoop::dispatch().
    RunLoopDispatcher = 100,

    // RunLoopTimer priority by default. It can be changed with RunLoopTimer::setPriority().
    RunLoopTimer = 0,

    // Garbage collector timers.
    JavascriptTimer = 200,

    // Memory pressure monitor.
    MemoryPressureHandlerTimer = -100,

    // WebCore timers.
    MainThreadSharedTimer = 100,

    // Used for timers that discard resources like backing store, buffers, etc.
    ReleaseUnusedResourcesTimer = 200,

    // Rendering timer in the threaded compositor.
    CompositingThreadUpdateTimer = 100,

    // Layer flush.
    LayerFlushTimer = 100,

    // DisplayRefreshMonitor timer, should have the same value as the LayerFlushTimer.
    DisplayRefreshMonitorTimer = 100,

    // Rendering timer in the main thread when accelerated compositing is not used.
    NonAcceleratedDrawingTimer = 100,

    // Async IO network callbacks.
    AsyncIONetwork = 100,
};

#else

enum RunLoopSourcePriority {
    RunLoopDispatcher = 0,
    RunLoopTimer = 0,

    MemoryPressureHandlerTimer = -10,

    JavascriptTimer = 10,
    MainThreadSharedTimer = 10,

    LayerFlushTimer = 0,
    DisplayRefreshMonitorTimer = 0,

    CompositingThreadUpdateTimer = 0,

    ReleaseUnusedResourcesTimer = 0,

    AsyncIONetwork = 10,
};

#endif

} // namespace WTF

using WTF::RunLoopSourcePriority;

#endif // USE(GLIB)
