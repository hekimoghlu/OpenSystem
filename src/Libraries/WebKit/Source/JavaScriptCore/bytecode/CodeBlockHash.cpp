/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
#include "CodeBlockHash.h"

#include "SourceCode.h"
#include <wtf/SHA1.h>
#include <wtf/SixCharacterHash.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

CodeBlockHash::CodeBlockHash(std::span<const char, 6> string)
    : m_hash(sixCharacterHashStringToInteger(string))
{
}

CodeBlockHash::CodeBlockHash(const SourceCode& sourceCode, CodeSpecializationKind kind)
    : m_hash(0)
{
    SHA1 sha1;

    // The maxSourceCodeLengthToHash is a heuristic to avoid crashing fuzzers
    // due to resource exhaustion. This is OK to do because:
    // 1. CodeBlockHash is not a critical hash.
    // 2. In practice, reasonable source code are not 500 MB or more long.
    // 3. And if they are that long, then we are still diversifying the hash on
    //    their length. But if they do collide, it's OK.
    // The only invariant here is that we should always produce the same hash
    // for the same source string. The algorithm below achieves that.
    ASSERT(sourceCode.length() >= 0);
    constexpr unsigned maxSourceCodeLengthToHash = 500 * MB;
    if (static_cast<unsigned>(sourceCode.length()) < maxSourceCodeLengthToHash)
        sha1.addUTF8Bytes(sourceCode.view());
    else {
        // Just hash with the length and samples of the source string instead.
        StringView str = sourceCode.provider()->source();
        unsigned index = 0;
        unsigned oldIndex = 0;
        unsigned length = str.length();
        unsigned step = (length >> 10) + 1;

        sha1.addBytes(std::span { std::bit_cast<uint8_t*>(&length), sizeof(length) });
        do {
            UChar character = str[index];
            sha1.addBytes(std::span { std::bit_cast<uint8_t*>(&character), sizeof(character) });
            oldIndex = index;
            index += step;
        } while (index > oldIndex && index < length);
    }

    SHA1::Digest digest;
    sha1.computeHash(digest);
    m_hash = digest[0] | (digest[1] << 8) | (digest[2] << 16) | (digest[3] << 24);

    if (m_hash == 0 || m_hash == 1)
        m_hash += 0x2d5a93d0; // Ensures a non-zero hash, and gets us #Azero0 for CodeForCall and #Azero1 for CodeForConstruct.
    static_assert(static_cast<unsigned>(CodeForCall) == 0);
    static_assert(static_cast<unsigned>(CodeForConstruct) == 1);
    m_hash ^= static_cast<unsigned>(kind);
    ASSERT(m_hash);
}

void CodeBlockHash::dump(PrintStream& out) const
{
    auto buffer = integerToSixCharacterHashString(m_hash);
    
#if ASSERT_ENABLED
    CodeBlockHash recompute(buffer);
    ASSERT(recompute == *this);
#endif // ASSERT_ENABLED
    
    out.print(std::span<const char> { buffer });
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
