/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 10, 2025.
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
#include "ParseHash.h"

#include "SourceCode.h"
#include <wtf/SHA1.h>

namespace JSC {

ParseHash::ParseHash(const SourceCode& sourceCode)
{
    SHA1 sha1;
    sha1.addUTF8Bytes(sourceCode.view());
    SHA1::Digest digest;
    sha1.computeHash(digest);
    unsigned hash = digest[0] | (digest[1] << 8) | (digest[2] << 16) | (digest[3] << 24);

    if (hash == 0 || hash == 1)
        hash += 0x2d5a93d0; // Ensures a non-zero hash, and gets us #Azero0 for CodeForCall and #Azero1 for CodeForConstruct.
    static_assert(static_cast<unsigned>(CodeForCall) == 0);
    static_assert(static_cast<unsigned>(CodeForConstruct) == 1);
    unsigned hashForCall = hash ^ static_cast<unsigned>(CodeForCall);
    unsigned hashForConstruct = hash ^ static_cast<unsigned>(CodeForConstruct);

    m_hashForCall = CodeBlockHash(hashForCall);
    m_hashForConstruct = CodeBlockHash(hashForConstruct);
    ASSERT(hashForCall);
    ASSERT(hashForConstruct);
}

} // namespace JSC

