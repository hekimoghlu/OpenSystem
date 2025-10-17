/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 24, 2025.
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

#include "VirtualRegister.h"

#include <wtf/Assertions.h>
#include <wtf/VectorTraits.h>

namespace JSC {

    class RegisterID {
        WTF_MAKE_NONCOPYABLE(RegisterID);

        friend class VirtualRegister;
    public:
        RegisterID()
            : m_refCount(0)
            , m_isTemporary(false)
#if ASSERT_ENABLED
            , m_didSetIndex(false)
#endif
        {
        }

        RegisterID(VirtualRegister virtualRegister)
            : m_refCount(0)
            , m_virtualRegister(virtualRegister)
            , m_isTemporary(false)
#if ASSERT_ENABLED
            , m_didSetIndex(true)
#endif
        {
        }
        
        explicit RegisterID(int index)
            : m_refCount(0)
            , m_virtualRegister(VirtualRegister(index))
            , m_isTemporary(false)
#if ASSERT_ENABLED
            , m_didSetIndex(true)
#endif
        {
        }

        void setIndex(VirtualRegister index)
        {
#if ASSERT_ENABLED
            m_didSetIndex = true;
#endif
            m_virtualRegister = index;
        }

        void setTemporary()
        {
            m_isTemporary = true;
        }

        int index() const
        {
            ASSERT(m_didSetIndex);
            return m_virtualRegister.offset();
        }

        VirtualRegister virtualRegister() const
        {
            ASSERT(m_virtualRegister.isValid());
            return m_virtualRegister;
        }

        bool isTemporary()
        {
            return m_isTemporary;
        }

        void ref()
        {
            ++m_refCount;
        }

        void deref()
        {
            --m_refCount;
            ASSERT(m_refCount >= 0);
        }

        int refCount() const
        {
            return m_refCount;
        }

    private:

        int m_refCount;
        VirtualRegister m_virtualRegister;
        bool m_isTemporary;
#if ASSERT_ENABLED
        bool m_didSetIndex;
#endif
    };
} // namespace JSC

namespace WTF {

    template<> struct VectorTraits<JSC::RegisterID> : VectorTraitsBase<true, JSC::RegisterID> {
        static constexpr bool needsInitialization = true;
        static constexpr bool canInitializeWithMemset = true; // Default initialization just sets everything to 0 or false, so this is safe.
    };

} // namespace WTF
