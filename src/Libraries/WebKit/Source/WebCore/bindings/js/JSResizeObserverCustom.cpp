/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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
#include "JSResizeObserver.h"

#include "Element.h"
#include "JSNodeCustom.h"
#include <JavaScriptCore/JSCInlines.h>

namespace WebCore {

template<typename Visitor>
void JSResizeObserver::visitAdditionalChildren(Visitor& visitor)
{
    ResizeObserverCallback* callback = wrapped().callbackConcurrently();
    if (callback)
        callback->visitJSFunction(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSResizeObserver);

bool JSResizeObserverOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, JSC::AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    if (JSC::jsCast<JSResizeObserver*>(handle.slot()->asCell())->wrapped().isReachableFromOpaqueRoots(visitor)) {
        if (UNLIKELY(reason))
            *reason = "Reachable from observed nodes"_s;
        return true;
    }
    return false;
}

} // namespace WebCore
