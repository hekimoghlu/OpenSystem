/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
#include "JSCanvasRenderingContext2D.h"

#include "JSNodeCustom.h"
#include "WebCoreOpaqueRootInlines.h"

namespace WebCore {

bool JSCanvasRenderingContext2DOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, JSC::AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    if (UNLIKELY(reason))
        *reason = "Canvas is opaque root"_s;

    JSCanvasRenderingContext2D* jsCanvasRenderingContext = JSC::jsCast<JSCanvasRenderingContext2D*>(handle.slot()->asCell());
    return containsWebCoreOpaqueRoot(visitor, jsCanvasRenderingContext->wrapped().canvas());
}

template<typename Visitor>
void JSCanvasRenderingContext2D::visitAdditionalChildren(Visitor& visitor)
{
    addWebCoreOpaqueRoot(visitor, wrapped().canvas());
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSCanvasRenderingContext2D);

} // namespace WebCore
