/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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

#include "CodeBlock.h"
#include "ObjectPropertyCondition.h"
#include "PackedCellPtr.h"
#include "Watchpoint.h"

namespace JSC { namespace DFG {

class AdaptiveStructureWatchpoint final : public Watchpoint {
public:
    AdaptiveStructureWatchpoint(const ObjectPropertyCondition&, CodeBlock*);
    AdaptiveStructureWatchpoint();
    
    const ObjectPropertyCondition& key() const { return m_key; }

    void initialize(const ObjectPropertyCondition&, CodeBlock*);

    void install(VM&);

    void fireInternal(VM&, const FireDetail&);

private:
    PackedCellPtr<CodeBlock> m_codeBlock;
    ObjectPropertyCondition m_key;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
