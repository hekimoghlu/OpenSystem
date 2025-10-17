/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "Fixups.h"

namespace mach_o {

const char* Fixup::keyName(uint8_t keyNum)
{
    assert(keyNum < 4);
    static const char* const names[] = {
        "IA", "IB", "DA", "DB"
    };
    return names[keyNum];
}

const char* Fixup::keyName() const
{
    assert(authenticated);
    return keyName(this->auth.key);
}

bool Fixup::operator==(const Fixup& other) const
{
    if ( location != other.location )
        return false;
    if ( segment != other.segment )
        return false;
    if ( authenticated != other.authenticated )
        return false;
    if ( authenticated ) {
        if ( auth.key != other.auth.key )
            return false;
        if ( auth.usesAddrDiversity != other.auth.usesAddrDiversity )
            return false;
        if ( auth.diversity != other.auth.diversity )
            return false;
    }
    if ( isBind != other.isBind )
        return false;
    if ( isBind ) {
        if ( bind.bindOrdinal != other.bind.bindOrdinal )
            return false;
        if ( bind.embeddedAddend != other.bind.embeddedAddend )
            return false;
    }
    else {
        if ( rebase.targetVmOffset != other.rebase.targetVmOffset )
            return false;
    }
    return true;
}


} // namespace mach_o
