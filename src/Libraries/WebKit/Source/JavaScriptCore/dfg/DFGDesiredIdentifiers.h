/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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

#if ENABLE(DFG_JIT)

#include "Identifier.h"
#include <wtf/HashMap.h>
#include <wtf/text/UniquedStringImpl.h>

namespace JSC {

class CodeBlock;
class VM;

namespace DFG {

class CommonData;

class DesiredIdentifiers {
public:
    DesiredIdentifiers();
    DesiredIdentifiers(CodeBlock*);
    ~DesiredIdentifiers();
    
    unsigned numberOfIdentifiers();
    unsigned ensure(UniquedStringImpl*);
    
    UniquedStringImpl* at(unsigned index) const;
    
    UniquedStringImpl* operator[](unsigned index) const { return at(index); }
    
    void reallyAdd(VM&, CommonData*);
    
private:
    void processCodeBlockIdentifiersIfNeeded();

    CodeBlock* m_codeBlock;
    Vector<UniquedStringImpl*> m_addedIdentifiers;
    UncheckedKeyHashMap<UniquedStringImpl*, unsigned> m_identifierNumberForName;
    bool m_didProcessIdentifiers;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
