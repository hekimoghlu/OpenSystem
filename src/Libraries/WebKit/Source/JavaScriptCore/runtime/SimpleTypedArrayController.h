/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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

#include "Handle.h"
#include "TypedArrayController.h"
#include "WeakHandleOwner.h"

namespace JSC {

// The default controller used for managing the relationship between
// array buffers and their wrappers in JavaScriptCore. This isn't what
// WebCore uses, but it is what JSC uses when running standalone. This
// is pretty simple:
//
// - If the JSArrayBuffer is live, then the ArrayBuffer stays alive.
//
// - If there is a JSArrayBufferView that is holding an ArrayBuffer
//   then any existing wrapper for that ArrayBuffer will be kept
//   alive.
//
// - If you ask an ArrayBuffer for a JSArrayBuffer after one had
//   already been created and it didn't die, then you get the same
//   one.

class SimpleTypedArrayController final : public TypedArrayController {
public:
    JS_EXPORT_PRIVATE SimpleTypedArrayController(bool allowAtomicsWait = true);
    ~SimpleTypedArrayController() final;
    
    JSArrayBuffer* toJS(JSGlobalObject*, JSGlobalObject*, ArrayBuffer*) final;
    void registerWrapper(JSGlobalObject*, ArrayBuffer*, JSArrayBuffer*) final;
    bool isAtomicsWaitAllowedOnCurrentThread() final;

private:
    class JSArrayBufferOwner final : public WeakHandleOwner {
    public:
        bool isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown>, void* context, AbstractSlotVisitor&, ASCIILiteral* reason) final;
        void finalize(JSC::Handle<JSC::Unknown>, void* context) final;
    };

    JSArrayBufferOwner m_owner;
    bool m_allowAtomicsWait { false };
};

} // namespace JSC
