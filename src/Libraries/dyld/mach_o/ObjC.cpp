/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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
#include <string.h>
#include <assert.h>

#include "ObjC.h"

namespace mach_o {

//
// MARK: --- ObjCMethodList methods ---
//

uint32_t ObjCMethodList::getMethodSize() const
{
    const uint32_t* header = (const uint32_t*)this;
    return header[0] & methodListSizeMask;
}

uint32_t ObjCMethodList::getMethodCount() const
{
    const uint32_t* header = (const uint32_t*)this;
    return header[1];
}

bool ObjCMethodList::usesRelativeOffsets() const
{
    const uint32_t* header = (const uint32_t*)this;
    return (header[0] & methodListIsRelative) != 0;
}

//
// MARK: --- ObjCProtocolList methods ---
//

uint32_t ObjCProtocolList::count(bool is64) const
{
    if ( is64 ) {
        const uint64_t* header = (const uint64_t*)this;
        return (uint32_t)header[0];
    }
    else {
        const uint32_t* header = (const uint32_t*)this;
        return header[0];
    }
}

//
// MARK: --- ObjCPropertyList methods ---
//

uint32_t ObjCPropertyList::getPropertySize() const
{
    const uint32_t* header = (const uint32_t*)this;
    return header[0];
}

uint32_t ObjCPropertyList::getPropertyCount() const
{
    const uint32_t* header = (const uint32_t*)this;
    return header[1];
}


} // namespace mach_o
