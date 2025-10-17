/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#include "config.h"
#include "B3DataSection.h"

#if ENABLE(B3_JIT)

#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace B3 {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DataSection);

DataSection::DataSection(size_t size)
    : m_data(fastZeroedMalloc(size))
    , m_size(size)
{
}

DataSection::~DataSection()
{
    fastFree(m_data);
}

void DataSection::dump(PrintStream& out) const
{
    out.print("DataSection at ", RawPointer(m_data), " with ", m_size, " bytes.");
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

