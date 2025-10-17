/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
#include "IdentifierRep.h"

#include "JSDOMBinding.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(IdentifierRep);

typedef UncheckedKeyHashSet<IdentifierRep*> IdentifierSet;

static IdentifierSet& identifierSet()
{
    static NeverDestroyed<IdentifierSet> identifierSet;
    return identifierSet;
}
    
using IntIdentifierMap = HashMap<int, IdentifierRep*>;

static IntIdentifierMap& intIdentifierMap()
{
    static NeverDestroyed<IntIdentifierMap> intIdentifierMap;
    return intIdentifierMap;
}

IdentifierRep* IdentifierRep::get(int intID)
{
    if (intID == 0 || intID == -1) {
        static NeverDestroyed<std::array<IdentifierRep*, 2>> negativeOneAndZeroIdentifiers;

        auto* identifier = negativeOneAndZeroIdentifiers.get()[intID + 1];
        if (!identifier) {
            identifier = new IdentifierRep(intID);

            negativeOneAndZeroIdentifiers.get()[intID + 1] = identifier;
        }
        
        return identifier;
    }
    
    IntIdentifierMap::AddResult result = intIdentifierMap().add(intID, nullptr);
    if (result.isNewEntry) {
        ASSERT(!result.iterator->value);
        result.iterator->value = new IdentifierRep(intID);
        
        identifierSet().add(result.iterator->value);
    }
    
    return result.iterator->value;
}

using StringIdentifierMap = HashMap<RefPtr<StringImpl>, IdentifierRep*>;

static StringIdentifierMap& stringIdentifierMap()
{
    static NeverDestroyed<StringIdentifierMap> stringIdentifierMap;
    return stringIdentifierMap;
}

IdentifierRep* IdentifierRep::get(const char* name)
{
    ASSERT(name);
    if (!name)
        return nullptr;
  
    String string = String::fromUTF8WithLatin1Fallback(unsafeSpan(name));
    StringIdentifierMap::AddResult result = stringIdentifierMap().add(string.impl(), nullptr);
    if (result.isNewEntry) {
        ASSERT(!result.iterator->value);
        result.iterator->value = new IdentifierRep(name);
        
        identifierSet().add(result.iterator->value);
    }
    
    return result.iterator->value;
}

bool IdentifierRep::isValid(IdentifierRep* identifier)
{
    return identifierSet().contains(identifier);
}
    
} // namespace WebCore
