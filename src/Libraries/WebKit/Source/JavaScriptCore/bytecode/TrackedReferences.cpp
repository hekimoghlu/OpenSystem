/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#include "TrackedReferences.h"

#include "JSCJSValueInlines.h"
#include <wtf/CommaPrinter.h>

namespace JSC {

TrackedReferences::TrackedReferences() = default;

TrackedReferences::~TrackedReferences() = default;

void TrackedReferences::add(JSCell* cell)
{
    if (cell)
        m_references.add(cell);
}

void TrackedReferences::add(JSValue value)
{
    if (value.isCell())
        add(value.asCell());
}

void TrackedReferences::check(JSCell* cell) const
{
    if (!cell)
        return;
    
    if (m_references.contains(cell))
        return;
    
    dataLog("Found untracked reference: ", JSValue(cell), "\n");
    dataLog("All tracked references: ", *this, "\n");
    RELEASE_ASSERT_NOT_REACHED();
}

void TrackedReferences::check(JSValue value) const
{
    if (value.isCell())
        check(value.asCell());
}

void TrackedReferences::dump(PrintStream& out) const
{
    CommaPrinter comma;
    for (JSCell* cell : m_references)
        out.print(comma, RawPointer(cell));
}

} // namespace JSC

