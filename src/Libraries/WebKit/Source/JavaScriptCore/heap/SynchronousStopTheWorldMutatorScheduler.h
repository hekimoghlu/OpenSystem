/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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

#include "MutatorScheduler.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

// The JSC concurrent GC relies on stopping the world to stay ahead of the retreating wavefront.
// It so happens that the same API can be reused to implement a non-concurrent GC mode, which we
// use on platforms that don't support the GC's atomicity protocols. That means anything other
// than X86-64 and ARM64. This scheduler is a drop-in replacement for the concurrent GC's
// SpaceTimeMutatorScheduler. It tells the GC to never resume the world once the GC cycle begins.

class SynchronousStopTheWorldMutatorScheduler final : public MutatorScheduler {
    WTF_MAKE_TZONE_ALLOCATED(SynchronousStopTheWorldMutatorScheduler);
public:
    SynchronousStopTheWorldMutatorScheduler();
    ~SynchronousStopTheWorldMutatorScheduler() final;
    
    State state() const final;
    
    void beginCollection() final;
    
    MonotonicTime timeToStop() final;
    MonotonicTime timeToResume() final;
    
    void endCollection() final;

private:
    State m_state { Normal };
};

} // namespace JSC

