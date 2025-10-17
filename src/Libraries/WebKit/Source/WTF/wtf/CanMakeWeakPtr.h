/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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

#include <wtf/CompactRefPtrTuple.h>
#include <wtf/Packed.h>
#include <wtf/WeakPtrFactory.h>
#include <wtf/WeakRef.h>

namespace WTF {

template<typename WeakPtrFactoryType, WeakPtrFactoryInitialization initializationMode = WeakPtrFactoryInitialization::Lazy>
class CanMakeWeakPtrBase {
public:
    using WeakValueType = typename WeakPtrFactoryType::ObjectType;
    using WeakPtrImplType = typename WeakPtrFactoryType::WeakPtrImplType;

    WeakPtrImplType* weakImplIfExists() const { return m_weakPtrFactory.impl(); }
    WeakPtrImplType& weakImpl() const
    {
        initializeWeakPtrFactory();
        return *m_weakPtrFactory.impl();
    }
    unsigned weakCount() const { return m_weakPtrFactory.weakPtrCount(); }

protected:
    CanMakeWeakPtrBase()
    {
        if (initializationMode == WeakPtrFactoryInitialization::Eager)
            initializeWeakPtrFactory();
    }

    CanMakeWeakPtrBase(const CanMakeWeakPtrBase&)
    {
        if (initializationMode == WeakPtrFactoryInitialization::Eager)
            initializeWeakPtrFactory();
    }

    CanMakeWeakPtrBase& operator=(const CanMakeWeakPtrBase&) { return *this; }

    void initializeWeakPtrFactory() const
    {
        m_weakPtrFactory.initializeIfNeeded(static_cast<const WeakValueType&>(*this));
    }

    const WeakPtrFactoryType& weakPtrFactory() const { return m_weakPtrFactory; }
    WeakPtrFactoryType& weakPtrFactory() { return m_weakPtrFactory; }

private:
    WeakPtrFactoryType m_weakPtrFactory;
};

template<typename T, WeakPtrFactoryInitialization initializationMode = WeakPtrFactoryInitialization::Lazy, typename WeakPtrImpl = DefaultWeakPtrImpl>
using CanMakeWeakPtr = CanMakeWeakPtrBase<WeakPtrFactory<T, WeakPtrImpl>, initializationMode>;

template<typename T, WeakPtrFactoryInitialization initializationMode = WeakPtrFactoryInitialization::Lazy, typename WeakPtrImpl = DefaultWeakPtrImpl>
using CanMakeWeakPtrWithBitField = CanMakeWeakPtrBase<WeakPtrFactoryWithBitField<T, WeakPtrImpl>, initializationMode>;

template<typename T, WeakPtrFactoryInitialization initializationMode = WeakPtrFactoryInitialization::Lazy>
using CanMakeSingleThreadWeakPtr = CanMakeWeakPtr<T, initializationMode, SingleThreadWeakPtrImpl>;

} // namespace WTF

using WTF::CanMakeWeakPtr;
using WTF::CanMakeWeakPtrWithBitField;
using WTF::CanMakeSingleThreadWeakPtr;
