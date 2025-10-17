/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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

namespace JSC {

class AbstractSlotVisitor;

class JS_EXPORT_PRIVATE WeakHandleOwner {
public:
    virtual ~WeakHandleOwner();
    // reason will only be non-null when generating a debug GC heap snapshot.
    virtual bool isReachableFromOpaqueRoots(Handle<Unknown>, void* context, AbstractSlotVisitor&, ASCIILiteral* reason = nullptr);
    virtual void finalize(Handle<Unknown>, void* context);
};

} // namespace JSC
