/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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

#include <JavaScriptCore/TypedArrayController.h>
#include <JavaScriptCore/WeakHandleOwner.h>

namespace JSC {
class WeakHandleOwner;
}

namespace WebCore {

class WebCoreTypedArrayController : public JSC::TypedArrayController {
public:
    WebCoreTypedArrayController(bool allowAtomicsWait);
    virtual ~WebCoreTypedArrayController();
    
    JSC::JSArrayBuffer* toJS(JSC::JSGlobalObject*, JSC::JSGlobalObject*, JSC::ArrayBuffer*) override;
    void registerWrapper(JSC::JSGlobalObject*, JSC::ArrayBuffer*, JSC::JSArrayBuffer*) override;
    bool isAtomicsWaitAllowedOnCurrentThread() override;

    JSC::WeakHandleOwner* wrapperOwner() { return &m_owner; }

private:
    class JSArrayBufferOwner : public JSC::WeakHandleOwner {
    public:
        bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, JSC::AbstractSlotVisitor&, ASCIILiteral*) override;
        void finalize(JSC::Handle<JSC::Unknown>, void* context) override;
    };

    JSArrayBufferOwner m_owner;
    bool m_allowAtomicsWait;
};

} // namespace WebCore
