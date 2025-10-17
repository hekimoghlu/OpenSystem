/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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
#ifndef SharedGDIObject_h
#define SharedGDIObject_h

#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/win/GDIObject.h>

namespace WebCore {

template <typename T> class SharedGDIObject : public RefCounted<SharedGDIObject<T>> {
public:
    static Ref<SharedGDIObject> create(GDIObject<T> object)
    {
        return adoptRef(*new SharedGDIObject<T>(WTFMove(object)));
    }

    T get() const
    {
        return m_object.get();
    }

    unsigned hash() const
    {
        return PtrHash<T>::hash(m_object.get());
    }

private:
    explicit SharedGDIObject(GDIObject<T> object)
        : m_object(WTFMove(object))
    {
    }

    GDIObject<T> m_object;
};

} // namespace WebCore

#endif // SharedGDIObject_h
