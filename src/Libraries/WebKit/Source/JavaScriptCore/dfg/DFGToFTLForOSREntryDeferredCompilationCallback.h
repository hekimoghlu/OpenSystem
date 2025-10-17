/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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

#if ENABLE(FTL_JIT)

#include "DFGJITCode.h"
#include "DeferredCompilationCallback.h"
#include <wtf/Ref.h>

namespace JSC {

class ScriptExecutable;

namespace DFG {

class ToFTLForOSREntryDeferredCompilationCallback final : public DeferredCompilationCallback {
public:
    ~ToFTLForOSREntryDeferredCompilationCallback() final;

    static Ref<ToFTLForOSREntryDeferredCompilationCallback> create(JITCode::TriggerReason* forcedOSREntryTrigger);

    void compilationDidBecomeReadyAsynchronously(CodeBlock*, CodeBlock* profiledDFGCodeBlock) final;
    void compilationDidComplete(CodeBlock*, CodeBlock* profiledDFGCodeBlock, CompilationResult) final;

private:
    ToFTLForOSREntryDeferredCompilationCallback(JITCode::TriggerReason* forcedOSREntryTrigger);

    JITCode::TriggerReason* m_forcedOSREntryTrigger;
};

} } // namespace JSC::DFG

#endif // ENABLE(FTL_JIT)
