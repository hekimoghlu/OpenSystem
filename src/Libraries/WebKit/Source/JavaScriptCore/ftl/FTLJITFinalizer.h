/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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

#include "DFGFinalizer.h"
#include "FTLJITCode.h"
#include "LinkBuffer.h"
#include "MacroAssembler.h"
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace FTL {

class JITFinalizer final : public DFG::Finalizer {
    WTF_MAKE_TZONE_ALLOCATED(JITFinalizer);
public:
    JITFinalizer(DFG::Plan&);
    ~JITFinalizer() final;

    size_t codeSize() final;
    bool finalize() final;
    bool isFailed() final { return false; };

    RefPtr<JSC::JITCode> jitCode() final { return m_jitCode; }

    RefPtr<FTL::JITCode> m_jitCode;
    size_t m_codeSize { 0 };
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
