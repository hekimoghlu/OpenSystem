/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include <wtf/Function.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class ISOProtectionSystemSpecificHeaderBox;
class SharedBuffer;

class InitDataRegistry {
public:
    WEBCORE_EXPORT static InitDataRegistry& shared();
    friend class NeverDestroyed<InitDataRegistry>;

    RefPtr<SharedBuffer> sanitizeInitData(const AtomString& initDataType, const SharedBuffer&);
    WEBCORE_EXPORT std::optional<Vector<Ref<SharedBuffer>>> extractKeyIDs(const AtomString& initDataType, const SharedBuffer&);

    struct InitDataTypeCallbacks {
        using SanitizeInitDataCallback = Function<RefPtr<SharedBuffer>(const SharedBuffer&)>;
        using ExtractKeyIDsCallback = Function<std::optional<Vector<Ref<SharedBuffer>>>(const SharedBuffer&)>;

        SanitizeInitDataCallback sanitizeInitData;
        ExtractKeyIDsCallback extractKeyIDs;
    };
    void registerInitDataType(const AtomString& initDataType, InitDataTypeCallbacks&&);

    static const AtomString& cencName();
    static const AtomString& keyidsName();
    static const AtomString& webmName();

    static std::optional<Vector<std::unique_ptr<ISOProtectionSystemSpecificHeaderBox>>> extractPsshBoxesFromCenc(const SharedBuffer&);
    static std::optional<Vector<Ref<SharedBuffer>>> extractKeyIDsCenc(const SharedBuffer&);
    static RefPtr<SharedBuffer> sanitizeCenc(const SharedBuffer&);

private:
    InitDataRegistry();
    ~InitDataRegistry();

    MemoryCompactRobinHoodHashMap<AtomString, InitDataTypeCallbacks> m_types;
};

}

#endif // ENABLE(ENCRYPTED_MEDIA)
