/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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

#include "JITPlan.h"

#if ENABLE(JIT)

#include "JIT.h"

namespace JSC {

class BaselineJITCode;

class BaselineJITPlan final : public JITPlan {
    using Base = JITPlan;

public:
    BaselineJITPlan(CodeBlock*);

    CompilationPath compileInThreadImpl() final;
    size_t codeSize() const final;
    CompilationResult finalize() override;

    CompilationPath compileSync(JITCompilationEffort);

    bool isKnownToBeLiveAfterGC() final;
    bool isKnownToBeLiveDuringGC(AbstractSlotVisitor&) final;

private:
    CompilationPath compileInThreadImpl(JITCompilationEffort);

    RefPtr<BaselineJITCode> m_jitCode;
};

} // namespace JSC

#endif // ENABLE(JIT)
