/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#include "JSNodeList.h"

#include "ChildNodeList.h"
#include "JSNode.h"
#include "LiveNodeList.h"
#include "Node.h"
#include "NodeList.h"
#include "WebCoreOpaqueRootInlines.h"
#include <wtf/text/AtomString.h>


namespace WebCore {
using namespace JSC;

bool JSNodeListOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    JSNodeList* jsNodeList = jsCast<JSNodeList*>(handle.slot()->asCell());
    if (!jsNodeList->hasCustomProperties())
        return false;

    if (jsNodeList->wrapped().isLiveNodeList()) {
        if (UNLIKELY(reason))
            *reason = "LiveNodeList owner is opaque root"_s;

        return containsWebCoreOpaqueRoot(visitor, static_cast<LiveNodeList&>(jsNodeList->wrapped()).ownerNode());
    }

    if (jsNodeList->wrapped().isChildNodeList()) {
        if (UNLIKELY(reason))
            *reason = "ChildNodeList owner is opaque root"_s;

        return containsWebCoreOpaqueRoot(visitor, static_cast<ChildNodeList&>(jsNodeList->wrapped()).ownerNode());
    }

    if (jsNodeList->wrapped().isEmptyNodeList()) {
        if (UNLIKELY(reason))
            *reason = "EmptyNodeList owner is opaque root"_s;

        return containsWebCoreOpaqueRoot(visitor, static_cast<EmptyNodeList&>(jsNodeList->wrapped()).ownerNode());
    }
    return false;
}

JSC::JSValue createWrapper(JSDOMGlobalObject& globalObject, Ref<NodeList>&& nodeList)
{
    // FIXME: Adopt reportExtraMemoryVisited, and switch to reportExtraMemoryAllocated.
    // https://bugs.webkit.org/show_bug.cgi?id=142595
    globalObject.vm().heap.deprecatedReportExtraMemory(nodeList->memoryCost());
    return createWrapper<NodeList>(&globalObject, WTFMove(nodeList));
}

JSC::JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<NodeList>&& nodeList)
{
    return createWrapper(*globalObject, WTFMove(nodeList));
}

} // namespace WebCore
