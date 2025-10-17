/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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

#include "InstructionStream.h"
#include <wtf/HashSet.h>

namespace JSC {

// Reference count indicates number of live registers that alias this object.
class StaticPropertyAnalysis : public RefCounted<StaticPropertyAnalysis> {
public:
    static Ref<StaticPropertyAnalysis> create(JSInstructionStream::MutableRef&& instructionRef)
    {
        return adoptRef(*new StaticPropertyAnalysis(WTFMove(instructionRef)));
    }

    void addPropertyIndex(unsigned propertyIndex) { m_propertyIndexes.add(propertyIndex); }

    void record();

    int propertyIndexCount() { return m_propertyIndexes.size(); }

private:
    StaticPropertyAnalysis(JSInstructionStream::MutableRef&& instructionRef)
        : m_instructionRef(WTFMove(instructionRef))
    {
    }

    JSInstructionStream::MutableRef m_instructionRef;
    typedef UncheckedKeyHashSet<unsigned, WTF::IntHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>> PropertyIndexSet;
    PropertyIndexSet m_propertyIndexes;
};

} // namespace JSC
