/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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

#include <wtf/Vector.h>
#include <wtf/text/ASCIILiteral.h>

#if ENABLE(IPC_TESTING_API)

namespace WebKit {

struct SerializedMember {
    ASCIILiteral type;
    ASCIILiteral name;
};

struct SerializedTypeInfo {
    ASCIILiteral name;
    Vector<SerializedMember> members;
};

Vector<SerializedTypeInfo> allSerializedTypes();

struct SerializedEnumInfo {
    ASCIILiteral name;
    uint32_t size;
    bool isOptionSet;
    Vector<uint64_t> validValues;
};

Vector<SerializedEnumInfo> allSerializedEnums();

}

#endif
