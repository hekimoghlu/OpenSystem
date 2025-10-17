/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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

#include "B3CaseCollection.h"
#include "B3SwitchValue.h"
#include "B3BasicBlock.h"

namespace JSC { namespace B3 {

inline const FrequentedBlock& CaseCollection::fallThrough() const
{
    return m_owner->fallThrough();
}

inline unsigned CaseCollection::size() const
{
    return m_switch->numCaseValues();
}

inline SwitchCase CaseCollection::at(unsigned index) const
{
    return SwitchCase(m_switch->caseValue(index), m_owner->successor(index));
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
