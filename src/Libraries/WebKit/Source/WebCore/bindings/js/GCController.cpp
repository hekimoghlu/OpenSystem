/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
#include "GCController.h"

#include "CommonVM.h"
#include "JSHTMLDocument.h"
#include "Location.h"
#include "WorkerGlobalScope.h"
#include <JavaScriptCore/Heap.h>
#include <JavaScriptCore/HeapSnapshotBuilder.h>
#include <JavaScriptCore/JSLock.h>
#include <JavaScriptCore/VM.h>
#include <pal/Logging.h>
#include <wtf/FileSystem.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;

static void collect()
{
    JSLockHolder lock(commonVM());
    commonVM().heap.collectNow(Async, CollectionScope::Full);
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(GCController);

GCController& GCController::singleton()
{
    static NeverDestroyed<GCController> controller;
    return controller;
}

GCController::GCController()
    : m_GCTimer(*this, &GCController::gcTimerFired)
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PAL::registerNotifyCallback("com.apple.WebKit.dumpGCHeap"_s, [] {
            GCController::singleton().dumpHeap();
        });
    });
}

void GCController::garbageCollectSoon()
{
    // We only use reportAbandonedObjectGraph for systems for which there's an implementation
    // of the garbage collector timers in JavaScriptCore. We wouldn't need this if JavaScriptCore
    // used a timer implementation from WTF like RunLoop::Timer.
#if USE(CF) || USE(GLIB)
    JSLockHolder lock(commonVM());
    commonVM().heap.reportAbandonedObjectGraph();
#else
    garbageCollectNow();
#endif
}

void GCController::garbageCollectOnNextRunLoop()
{
    if (!m_GCTimer.isActive())
        m_GCTimer.startOneShot(0_s);
}

void GCController::gcTimerFired()
{
    collect();
}

void GCController::garbageCollectNow()
{
    JSLockHolder lock(commonVM());
    if (!commonVM().heap.currentThreadIsDoingGCWork()) {
        commonVM().heap.collectNow(Sync, CollectionScope::Full);
        WTF::releaseFastMallocFreeMemory();
    }
}

void GCController::garbageCollectNowIfNotDoneRecently()
{
#if USE(CF) || USE(GLIB)
    JSLockHolder lock(commonVM());
    if (!commonVM().heap.currentThreadIsDoingGCWork())
        commonVM().heap.collectNowFullIfNotDoneRecently(Async);
#else
    garbageCollectNow();
#endif
}

void GCController::garbageCollectOnAlternateThreadForDebugging(bool waitUntilDone)
{
    auto thread = Thread::create("WebCore: GCController"_s, &collect, ThreadType::GarbageCollection);

    if (waitUntilDone) {
        thread->waitForCompletion();
        return;
    }

    thread->detach();
}

void GCController::setJavaScriptGarbageCollectorTimerEnabled(bool enable)
{
    commonVM().heap.setGarbageCollectionTimerEnabled(enable);
}

void GCController::deleteAllCode(DeleteAllCodeEffort effort)
{
    JSLockHolder lock(commonVM());
    commonVM().deleteAllCode(effort);
}

void GCController::deleteAllLinkedCode(DeleteAllCodeEffort effort)
{
    JSLockHolder lock(commonVM());
    commonVM().deleteAllLinkedCode(effort);
}

void GCController::dumpHeapForVM(VM& vm)
{
    auto [tempFilePath, fileHandle] = FileSystem::openTemporaryFile("GCHeap"_s);
    if (!FileSystem::isHandleValid(fileHandle)) {
        WTFLogAlways("Dumping GC heap failed to open temporary file");
        return;
    }

    JSLockHolder lock(vm);
    sanitizeStackForVM(vm);

    String jsonData;
    {
        DeferGCForAWhile deferGC(vm); // Prevent concurrent GC from interfering with the full GC that the snapshot does.

        HeapSnapshotBuilder snapshotBuilder(vm.ensureHeapProfiler(), HeapSnapshotBuilder::SnapshotType::GCDebuggingSnapshot);
        snapshotBuilder.buildSnapshot();

        jsonData = snapshotBuilder.json();
    }

    CString utf8String = jsonData.utf8();

    FileSystem::writeToFile(fileHandle, utf8String.span());
    FileSystem::closeFile(fileHandle);
    WTFLogAlways("Dumped GC heap to %s%s", tempFilePath.utf8().data(), isMainThread() ? "" : " for Worker");
}

void GCController::dumpHeap()
{
    dumpHeapForVM(commonVM());
    WorkerGlobalScope::dumpGCHeapForWorkers();
}

} // namespace WebCore
