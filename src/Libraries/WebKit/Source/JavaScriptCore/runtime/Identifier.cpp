/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#include "Identifier.h"

#include "IdentifierInlines.h"
#include "NumericStrings.h"
#include <wtf/Assertions.h>

namespace JSC {

Ref<AtomStringImpl> Identifier::add8(VM& vm, std::span<const UChar> s)
{
    if (s.size() == 1) {
        UChar c = s.front();
        ASSERT(isLatin1(c));
        if (canUseSingleCharacterString(c))
            return vm.smallStrings.singleCharacterStringRep(c);
    }
    if (s.empty())
        return *static_cast<AtomStringImpl*>(StringImpl::empty());
    return *AtomStringImpl::add(s);
}

Identifier Identifier::from(VM& vm, unsigned value)
{
    return Identifier(vm, vm.numericStrings.add(value));
}

Identifier Identifier::from(VM& vm, int value)
{
    return Identifier(vm, vm.numericStrings.add(value));
}

Identifier Identifier::from(VM& vm, double value)
{
    return Identifier(vm, vm.numericStrings.add(value));
}

void Identifier::dump(PrintStream& out) const
{
    if (impl()) {
        if (impl()->isSymbol()) {
            auto* symbol = static_cast<SymbolImpl*>(impl());
            if (symbol->isPrivate())
                out.print("PrivateSymbol.");
        }
        out.print(impl());
    } else
        out.print("<null identifier>");
}

#ifndef NDEBUG

void Identifier::checkCurrentAtomStringTable(VM& vm)
{
    // Check the identifier table accessible through the threadspecific matches the
    // vm's identifier table.
    ASSERT_UNUSED(vm, vm.atomStringTable() == Thread::current().atomStringTable());
}

#else

// These only exists so that our exports are the same for debug and release builds.
// This would be an RELEASE_ASSERT_NOT_REACHED(), but we're in NDEBUG only code here!
NO_RETURN_DUE_TO_CRASH void Identifier::checkCurrentAtomStringTable(VM&) { CRASH(); }

#endif

} // namespace JSC
