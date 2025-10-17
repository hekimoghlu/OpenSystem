/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
#include "DFGFinalizer.h"
#include "JITCode.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(DFG_JIT)

namespace JSC { namespace DFG {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Finalizer);

Finalizer::Finalizer(Plan& plan)
    : m_plan(plan)
{
}

Finalizer::~Finalizer() = default;


RefPtr<JSC::JITCode> Finalizer::jitCode()
{
    return nullptr;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

