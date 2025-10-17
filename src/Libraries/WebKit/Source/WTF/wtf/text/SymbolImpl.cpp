/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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
#include <wtf/text/SymbolImpl.h>

namespace WTF {

// In addition to the normal hash value, store specialized hash value for
// symbolized StringImpl*. And don't use the normal hash value for symbolized
// StringImpl* when they are treated as Identifiers. Unique nature of these
// symbolized StringImpl* keys means that we don't need them to match any other
// string (in fact, that's exactly the oposite of what we want!), and the
// normal hash would lead to lots of conflicts.
unsigned SymbolImpl::nextHashForSymbol()
{
    static unsigned s_nextHashForSymbol = 0;
    s_nextHashForSymbol += 1 << s_flagCount;
    s_nextHashForSymbol |= 1u << 31;
    return s_nextHashForSymbol;
}

Ref<SymbolImpl> SymbolImpl::create(StringImpl& rep)
{
    auto* ownerRep = (rep.bufferOwnership() == BufferSubstring) ? rep.substringBuffer() : &rep;
    ASSERT(ownerRep->bufferOwnership() != BufferSubstring);
    if (rep.is8Bit())
        return adoptRef(*new SymbolImpl(rep.span8(), *ownerRep));
    return adoptRef(*new SymbolImpl(rep.span16(), *ownerRep));
}

Ref<SymbolImpl> SymbolImpl::createNullSymbol()
{
    return adoptRef(*new SymbolImpl);
}

Ref<PrivateSymbolImpl> PrivateSymbolImpl::create(StringImpl& rep)
{
    auto* ownerRep = (rep.bufferOwnership() == BufferSubstring) ? rep.substringBuffer() : &rep;
    ASSERT(ownerRep->bufferOwnership() != BufferSubstring);
    if (rep.is8Bit())
        return adoptRef(*new PrivateSymbolImpl(rep.span8(), *ownerRep));
    return adoptRef(*new PrivateSymbolImpl(rep.span16(), *ownerRep));
}

Ref<RegisteredSymbolImpl> RegisteredSymbolImpl::create(StringImpl& rep, SymbolRegistry& symbolRegistry)
{
    auto* ownerRep = (rep.bufferOwnership() == BufferSubstring) ? rep.substringBuffer() : &rep;
    ASSERT(ownerRep->bufferOwnership() != BufferSubstring);
    if (rep.is8Bit())
        return adoptRef(*new RegisteredSymbolImpl(rep.span8(), *ownerRep, symbolRegistry));
    return adoptRef(*new RegisteredSymbolImpl(rep.span16(), *ownerRep, symbolRegistry));
}

Ref<RegisteredSymbolImpl> RegisteredSymbolImpl::createPrivate(StringImpl& rep, SymbolRegistry& symbolRegistry)
{
    auto* ownerRep = (rep.bufferOwnership() == BufferSubstring) ? rep.substringBuffer() : &rep;
    ASSERT(ownerRep->bufferOwnership() != BufferSubstring);
    if (rep.is8Bit())
        return adoptRef(*new RegisteredSymbolImpl(rep.span8(), *ownerRep, symbolRegistry, s_flagIsRegistered | s_flagIsPrivate));
    return adoptRef(*new RegisteredSymbolImpl(rep.span16(), *ownerRep, symbolRegistry, s_flagIsRegistered | s_flagIsPrivate));
}

} // namespace WTF
