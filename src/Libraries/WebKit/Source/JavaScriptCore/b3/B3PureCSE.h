/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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

#include "B3ValueKey.h"
#include <wtf/HashMap.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

class BasicBlock;
class Dominators;
class Value;

typedef Vector<Value*, 1> Matches;

// This is a reusable utility for doing pure CSE. You can use it to do pure CSE on a program by just
// proceeding in order and calling process().
class PureCSE {
public:
    PureCSE();
    ~PureCSE();

    void clear();

    Value* findMatch(const ValueKey&, BasicBlock*, Dominators&);

    bool process(Value*, Dominators&);
    
private:
    UncheckedKeyHashMap<ValueKey, Matches> m_map;
};

bool pureCSE(Procedure&);

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
