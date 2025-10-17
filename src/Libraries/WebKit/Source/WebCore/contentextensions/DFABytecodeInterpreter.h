/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 27, 2025.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "ContentExtensionRule.h"
#include "ContentExtensionsDebugging.h"
#include "DFABytecode.h"
#include <wtf/DataLog.h>
#include <wtf/HashSet.h>

namespace WebCore::ContentExtensions {

class DFABytecodeInterpreter {
public:
    DFABytecodeInterpreter(std::span<const uint8_t> bytecode)
        : m_bytecode(bytecode) { }

    using Actions = UncheckedKeyHashSet<uint64_t, DefaultHash<uint64_t>, WTF::UnsignedWithZeroKeyHashTraits<uint64_t>>;

    WEBCORE_EXPORT Actions interpret(const String&, ResourceFlags);
    WEBCORE_EXPORT Actions actionsMatchingEverything();

private:
    void interpretAppendAction(unsigned& programCounter, Actions&);
    void interpretTestFlagsAndAppendAction(unsigned& programCounter, ResourceFlags, Actions&);

    template<bool caseSensitive>
    void interpretJumpTable(std::span<const LChar> url, uint32_t& urlIndex, uint32_t& programCounter);

    const std::span<const uint8_t> m_bytecode;
};

} // namespace WebCore::ContentExtensions

#endif // ENABLE(CONTENT_EXTENSIONS)
