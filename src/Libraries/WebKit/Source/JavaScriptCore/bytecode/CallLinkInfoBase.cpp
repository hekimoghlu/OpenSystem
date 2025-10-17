/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#include "CallLinkInfoBase.h"

#include "CachedCall.h"
#include "CallLinkInfo.h"
#include "JSCJSValueInlines.h"
#include "PolymorphicCallStubRoutine.h"

namespace JSC {

void CallLinkInfoBase::unlinkOrUpgrade(VM& vm, CodeBlock* oldCodeBlock, CodeBlock* newCodeBlock)
{
    switch (callSiteType()) {
    case CallSiteType::CallLinkInfo:
        static_cast<CallLinkInfo*>(this)->unlinkOrUpgradeImpl(vm, oldCodeBlock, newCodeBlock);
        break;
    case CallSiteType::PolymorphicCallNode:
        static_cast<PolymorphicCallNode*>(this)->unlinkOrUpgradeImpl(vm, oldCodeBlock, newCodeBlock);
        break;
#if ENABLE(JIT)
    case CallSiteType::DirectCall:
        static_cast<DirectCallLinkInfo*>(this)->unlinkOrUpgradeImpl(vm, oldCodeBlock, newCodeBlock);
        break;
#endif
    case CallSiteType::CachedCall:
        static_cast<CachedCall*>(this)->unlinkOrUpgradeImpl(vm, oldCodeBlock, newCodeBlock);
        break;
    }
}

} // namespace JSC
