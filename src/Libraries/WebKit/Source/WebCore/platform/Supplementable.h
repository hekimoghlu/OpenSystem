/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#ifndef Supplementable_h
#define Supplementable_h

#include <wtf/Assertions.h>
#include <wtf/HashMap.h>
#include <wtf/MainThread.h>
#include <wtf/text/ASCIILiteral.h>

#if ASSERT_ENABLED
#include <wtf/Threading.h>
#endif

namespace WebCore {

// What you should know about Supplementable and Supplement
// ========================================================
// Supplementable and Supplement instances are meant to be thread local. They
// should only be accessed from within the thread that created them. The
// 2 classes are not designed for safe access from another thread. Violating
// this design assumption can result in memory corruption and unpredictable
// behavior.
//
// What you should know about the Supplement keys
// ==============================================
// The Supplement is expected to use the same ASCIILiteral instance as its
// key. The Supplementable's SupplementMap will use the address of the
// string as the key and not the characters themselves. Hence, 2 strings with
// the same characters will be treated as 2 different keys.
//
// In practice, it is recommended that Supplements implements a static method
// for returning its key to use. For example:
//
//     class MyClass : public Supplement<MySupplementable> {
//         ...
//         static ASCIILiteral supplementName();
//     }
//
//     ASCIILiteral MyClass::supplementName()
//     {
//         return "MyClass"_s;
//     }
//
// An example of the using the key:
//
//     MyClass* MyClass::from(MySupplementable* host)
//     {
//         return reinterpret_cast<MyClass*>(Supplement<MySupplementable>::from(host, supplementName()));
//     }

template<typename T>
class Supplementable;

template<typename T>
class Supplement {
public:
    virtual ~Supplement() = default;
#if ASSERT_ENABLED || ENABLE(SECURITY_ASSERTIONS)
    virtual bool isRefCountedWrapper() const { return false; }
#endif

    static void provideTo(Supplementable<T>* host, ASCIILiteral key, std::unique_ptr<Supplement<T>> supplement)
    {
        host->provideSupplement(key, WTFMove(supplement));
    }

    static Supplement<T>* from(Supplementable<T>* host, ASCIILiteral key)
    {
        return host ? host->requireSupplement(key) : 0;
    }
};

template<typename T>
class Supplementable {
public:
    void provideSupplement(ASCIILiteral key, std::unique_ptr<Supplement<T>> supplement)
    {
        ASSERT(canCurrentThreadAccessThreadLocalData(m_thread));
        ASSERT(!m_supplements.get(key));
        m_supplements.add(key, WTFMove(supplement));
    }

    Supplement<T>* requireSupplement(ASCIILiteral key)
    {
        ASSERT(canCurrentThreadAccessThreadLocalData(m_thread));
        return m_supplements.get(key);
    }

#if ASSERT_ENABLED
protected:
    Supplementable() = default;
#endif

private:
    using SupplementMap = UncheckedKeyHashMap<ASCIILiteral, std::unique_ptr<Supplement<T>>>;
    SupplementMap m_supplements;
#if ASSERT_ENABLED
    Ref<Thread> m_thread { Thread::current() };
#endif
};

} // namespace WebCore

#endif // Supplementable_h

