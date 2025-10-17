/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

#include "DFGCommon.h"

#if ENABLE(FTL_JIT)

#include "CCallHelpers.h"
#include "FTLOSRExitHandle.h"
#include <wtf/Box.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace JSC { namespace FTL {

class State;

class ExceptionTarget : public ThreadSafeRefCounted<ExceptionTarget> {
public:
    ~ExceptionTarget();

    // It's OK to call this during linking, but not any sooner.
    CodeLocationLabel<ExceptionHandlerPtrTag> label(LinkBuffer&);

    // Or, you can get a JumpList at any time. Anything you add to this JumpList will be linked to
    // the target's label.
    Box<CCallHelpers::JumpList> jumps(CCallHelpers&);
    
private:
    friend class PatchpointExceptionHandle;

    ExceptionTarget(bool isDefaultHandler, Box<CCallHelpers::Label>, RefPtr<OSRExitHandle>);
    
    bool m_isDefaultHandler;
    Box<CCallHelpers::Label> m_defaultHandler;
    RefPtr<OSRExitHandle> m_handle;
};

} } // namespace JSC::FTL

#endif // ENABLE(FTL_JIT)
