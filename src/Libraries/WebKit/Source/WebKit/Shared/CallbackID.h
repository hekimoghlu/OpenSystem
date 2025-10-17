/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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

#include <wtf/ArgumentCoder.h>
#include <wtf/HashTraits.h>
#include <wtf/RunLoop.h>

namespace WTF {

struct CallbackIDHash;

}

namespace WebKit {

class CallbackID {
public:
    ALWAYS_INLINE explicit CallbackID()
    { }

    ALWAYS_INLINE CallbackID(const CallbackID& otherID)
        : m_id(otherID.m_id)
    {
        ASSERT(HashTraits<uint64_t>::emptyValue() != m_id && !HashTraits<uint64_t>::isDeletedValue(m_id));
    }

    ALWAYS_INLINE CallbackID& operator=(const CallbackID& otherID)
    {
        m_id = otherID.m_id;
        return *this;
    }

    bool operator==(const CallbackID& other) const { return m_id == other.m_id; }

    uint64_t toInteger() const { return m_id; }
    ALWAYS_INLINE bool isValid() const { return isValidCallbackID(m_id); }
    static ALWAYS_INLINE bool isValidCallbackID(uint64_t rawId)
    {
        return HashTraits<uint64_t>::emptyValue() != rawId && !HashTraits<uint64_t>::isDeletedValue(rawId);
    }

    static ALWAYS_INLINE CallbackID fromInteger(uint64_t rawId)
    {
        RELEASE_ASSERT(isValidCallbackID(rawId));
        return CallbackID(rawId);
    }

    static CallbackID generateID()
    {
        ASSERT(RunLoop::isMain());
        static uint64_t uniqueCallbackID = 1;
        return CallbackID(uniqueCallbackID++);
    }

private:
    friend struct IPC::ArgumentCoder<CallbackID, void>;
    ALWAYS_INLINE explicit CallbackID(uint64_t newID)
        : m_id(newID)
    {
        ASSERT(newID != HashTraits<uint64_t>::emptyValue());
    }

    friend class CallbackMap;
    template <typename CallbackType> friend class SpecificCallbackMap;
    friend struct WTF::CallbackIDHash;
    friend HashTraits<WebKit::CallbackID>;

    uint64_t m_id { HashTraits<uint64_t>::emptyValue() };
};

}

namespace WTF {

struct CallbackIDHash {
    static unsigned hash(const WebKit::CallbackID& callbackID) { return intHash(callbackID.m_id); }
    static bool equal(const WebKit::CallbackID& a, const WebKit::CallbackID& b) { return a.m_id == b.m_id; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};
template<> struct HashTraits<WebKit::CallbackID> : GenericHashTraits<WebKit::CallbackID> {
    static WebKit::CallbackID emptyValue() { return WebKit::CallbackID(); }
    static void constructDeletedValue(WebKit::CallbackID& slot) { HashTraits<uint64_t>::constructDeletedValue(slot.m_id); }
    static bool isDeletedValue(const WebKit::CallbackID& slot) { return HashTraits<uint64_t>::isDeletedValue(slot.m_id); }
};
template<> struct DefaultHash<WebKit::CallbackID> : CallbackIDHash { };

}
