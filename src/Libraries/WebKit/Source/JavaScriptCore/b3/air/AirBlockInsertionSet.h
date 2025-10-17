/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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

#if ENABLE(B3_JIT)

#include "B3GenericBlockInsertionSet.h"
#include "AirCode.h"
#include <wtf/Insertion.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 { namespace Air {

class InsertionSet;

typedef GenericBlockInsertionSet<BasicBlock>::BlockInsertion BlockInsertion;

class BlockInsertionSet : public GenericBlockInsertionSet<BasicBlock> {
public:
    BlockInsertionSet(Code&);
    ~BlockInsertionSet();
    
    // FIXME: We should eventually implement B3::BlockInsertionSet's splitForward().
    // https://bugs.webkit.org/show_bug.cgi?id=169253
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
