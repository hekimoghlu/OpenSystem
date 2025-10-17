/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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

#include "HeapCellType.h"
#include <wtf/PtrTag.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class JS_EXPORT_PRIVATE IsoHeapCellType final : public HeapCellType {
    WTF_MAKE_TZONE_ALLOCATED(IsoHeapCellType);
public:
    using DestroyFunctionPtr = void (*)(JSCell*);

    ~IsoHeapCellType();

    template<typename CellType>
    struct Args {
        Args()
            : mode(CellType::needsDestruction)
            , functionPtr(&CellType::destroy)
        { }

        DestructionMode mode;
        DestroyFunctionPtr functionPtr;
    };

    template<typename CellType>
    IsoHeapCellType(Args<CellType>&& args)
        : IsoHeapCellType(args.mode, args.functionPtr)
    { }

    void finishSweep(MarkedBlock::Handle&, FreeList*) const final;
    void destroy(VM&, JSCell*) const final;

    ALWAYS_INLINE void operator()(VM&, JSCell* cell) const
    {
        m_destroy(cell);
    }

private:
    IsoHeapCellType(DestructionMode, DestroyFunctionPtr);

    DestroyFunctionPtr WTF_VTBL_FUNCPTR_PTRAUTH_STR("IsoHeapCellType.destroy") m_destroy;
};

} // namespace JSC
