/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#include "JSAbortSignal.h"

#include "WebCoreOpaqueRootInlines.h"

namespace WebCore {

bool JSAbortSignalOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, JSC::AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    auto& abortSignal = JSC::jsCast<JSAbortSignal*>(handle.slot()->asCell())->wrapped();
    if (abortSignal.aborted())
        return false;

    if (abortSignal.isFollowingSignal()) {
        if (UNLIKELY(reason))
            *reason = "Is Following Signal"_s;
        return true;
    }

    if (abortSignal.hasAbortEventListener()) {
        if (abortSignal.hasActiveTimeoutTimer()) {
            if (UNLIKELY(reason))
                *reason = "Has Timeout And Abort Event Listener"_s;
            return true;
        }
        if (abortSignal.isDependent()) {
            if (!abortSignal.sourceSignals().isEmptyIgnoringNullReferences()) {
                if (UNLIKELY(reason))
                    *reason = "Has Source Signals And Abort Event Listener"_s;
                return true;
            }
        } else {
            if (UNLIKELY(reason))
                *reason = "Has Abort Event Listener"_s;
            return true;
        }
    }

    return containsWebCoreOpaqueRoot(visitor, abortSignal);
}

template<typename Visitor>
void JSAbortSignal::visitAdditionalChildren(Visitor& visitor)
{
    wrapped().reason().visit(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSAbortSignal);

} // namespace WebCore
