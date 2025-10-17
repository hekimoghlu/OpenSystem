/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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

#include "WorkerOrWorkletGlobalScope.h"
#include <JavaScriptCore/Debugger.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WorkerDebugger final : public JSC::Debugger {
    WTF_MAKE_NONCOPYABLE(WorkerDebugger);
    WTF_MAKE_TZONE_ALLOCATED(WorkerDebugger);
public:
    WorkerDebugger(WorkerOrWorkletGlobalScope&);
    ~WorkerDebugger() override = default;


private:
    // JSC::Debugger
    void attachDebugger() final;
    void detachDebugger(bool isBeingDestroyed) final;
    void recompileAllJSFunctions() final;
    void runEventLoopWhilePaused() final;
    void reportException(JSC::JSGlobalObject*, JSC::Exception*) const final;

    WeakRef<WorkerOrWorkletGlobalScope> m_globalScope;
};

} // namespace WebCore
