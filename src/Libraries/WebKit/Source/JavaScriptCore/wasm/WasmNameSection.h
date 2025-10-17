/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

#include "WasmName.h"
#include <wtf/Noncopyable.h>
#include <wtf/text/CString.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>
#include <utility>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace Wasm {

struct NameSection : public ThreadSafeRefCounted<NameSection> {
    WTF_MAKE_NONCOPYABLE(NameSection);
public:
    NameSection()
    {
        setHash(std::nullopt);
    }

    static Ref<NameSection> create()
    {
        return adoptRef(*new NameSection);
    }

    void setHash(const std::optional<CString> &hash)
    {
        moduleHash = Name(hash ? hash->length() : 3);
        if (hash) {
            for (size_t i = 0; i < hash->length(); ++i)
                moduleHash[i] = static_cast<uint8_t>(*(hash->data() + i));
        } else {
            moduleHash[0] = '<';
            moduleHash[1] = '?';
            moduleHash[2] = '>';
        }
    }

    std::pair<const Name*, RefPtr<NameSection>> get(size_t functionIndexSpace)
    {
        return std::make_pair(functionIndexSpace < functionNames.size() ? &functionNames[functionIndexSpace] : nullptr, RefPtr { this });
    }
    Name moduleName;
    Name moduleHash;
    Vector<Name> functionNames;
};

} } // namespace JSC::Wasm

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
