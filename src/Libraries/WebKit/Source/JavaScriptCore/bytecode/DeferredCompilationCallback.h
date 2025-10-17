/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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

#include "CompilationResult.h"
#include "DeferredSourceDump.h"
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace JSC {

class CodeBlock;

class DeferredCompilationCallback : public RefCounted<DeferredCompilationCallback> {
protected:
    DeferredCompilationCallback();

public:
    virtual ~DeferredCompilationCallback();

    virtual void compilationDidBecomeReadyAsynchronously(CodeBlock*, CodeBlock* profiledDFGCodeBlock) = 0;
    virtual void compilationDidComplete(CodeBlock*, CodeBlock* profiledDFGCodeBlock, CompilationResult);

    Vector<DeferredSourceDump>& ensureDeferredSourceDump();

private:
    void dumpCompiledSourcesIfNeeded();

    std::unique_ptr<Vector<DeferredSourceDump>> m_deferredSourceDump;
};

} // namespace JSC
