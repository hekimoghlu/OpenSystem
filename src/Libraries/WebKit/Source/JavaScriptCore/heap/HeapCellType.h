/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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

#include "MarkedBlock.h"
#include <wtf/ForbidHeapAllocation.h>

namespace JSC {

class HeapCellType {
    WTF_MAKE_NONCOPYABLE(HeapCellType);
    WTF_FORBID_HEAP_ALLOCATION;
public:
    JS_EXPORT_PRIVATE HeapCellType(CellAttributes);
    JS_EXPORT_PRIVATE virtual ~HeapCellType();

    CellAttributes attributes() const { return m_attributes; }

    // The purpose of overriding this is to specialize the sweep for your destructors. This won't
    // be called for no-destructor blocks. This must call MarkedBlock::finishSweepKnowingSubspace.
    virtual void finishSweep(MarkedBlock::Handle&, FreeList*) const;

    // These get called for large objects.
    virtual void destroy(VM&, JSCell*) const;

private:
    CellAttributes m_attributes;
};

} // namespace JSC

