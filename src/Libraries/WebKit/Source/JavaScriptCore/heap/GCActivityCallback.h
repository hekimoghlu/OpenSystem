/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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

#include "JSRunLoopTimer.h"
#include "Synchronousness.h"
#include <wtf/RefPtr.h>

namespace JSC {

class FullGCActivityCallback;
class Heap;

class GCActivityCallback : public JSRunLoopTimer {
public:
    using Base = JSRunLoopTimer;

    JS_EXPORT_PRIVATE GCActivityCallback(Heap&, Synchronousness);
    JS_EXPORT_PRIVATE ~GCActivityCallback();

    JS_EXPORT_PRIVATE void doWork(VM&) override;

    virtual void doCollection(VM&) = 0;

    void didAllocate(Heap&, size_t);
    void willCollect();
    JS_EXPORT_PRIVATE void cancel();
    bool isEnabled() const { return m_enabled; }
    void setEnabled(bool enabled) { m_enabled = enabled; }
    bool didGCRecently() const { return m_didGCRecently; }
    void setDidGCRecently(bool didGCRecently) { m_didGCRecently = didGCRecently; }

    static bool s_shouldCreateGCTimer;

protected:
    virtual Seconds lastGCLength(Heap&) = 0;
    virtual double gcTimeSlice(size_t bytes) = 0;
    virtual double deathRate(Heap&) = 0;
    JS_EXPORT_PRIVATE void scheduleTimer(Seconds);

    GCActivityCallback(VM&, Synchronousness);

    Synchronousness m_synchronousness { Synchronousness::Async };
    bool m_enabled { true };
    bool m_didGCRecently { false };
    Seconds m_delay { s_decade };
};

} // namespace JSC
