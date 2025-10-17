/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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

#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class CachedResource;

class CachedResourceHandleBase {
public:
    WEBCORE_EXPORT ~CachedResourceHandleBase();

    WEBCORE_EXPORT CachedResource* get() const;
    
    bool operator!() const { return !m_resource; }
    operator bool() const { return !!m_resource; }

protected:
    WEBCORE_EXPORT CachedResourceHandleBase();
    WEBCORE_EXPORT explicit CachedResourceHandleBase(CachedResource*);
    WEBCORE_EXPORT explicit CachedResourceHandleBase(CachedResource&);
    WEBCORE_EXPORT CachedResourceHandleBase(const CachedResourceHandleBase&);

    WEBCORE_EXPORT void setResource(CachedResource*);
    
private:
    CachedResourceHandleBase& operator=(const CachedResourceHandleBase&) { return *this; } 
    
    friend class CachedResource;

    WeakPtr<CachedResource> m_resource;
};
    
template <class R> class CachedResourceHandle : public CachedResourceHandleBase {
public: 
    CachedResourceHandle() = default;
    CachedResourceHandle(R& res) : CachedResourceHandleBase(res) { }
    CachedResourceHandle(R* res) : CachedResourceHandleBase(res) { }
    CachedResourceHandle(const CachedResourceHandle<R>& o) : CachedResourceHandleBase(o) { }
    template<typename U> CachedResourceHandle(const CachedResourceHandle<U>& o) : CachedResourceHandleBase(o.get()) { }

    R* get() const { return reinterpret_cast<R*>(CachedResourceHandleBase::get()); }
    R* operator->() const { return get(); }
    R& operator*() const { ASSERT(get()); return *get(); }

    CachedResourceHandle& operator=(R* res) { setResource(res); return *this; } 
    CachedResourceHandle& operator=(const CachedResourceHandle& o) { setResource(o.get()); return *this; }
    template<typename U> CachedResourceHandle& operator=(const CachedResourceHandle<U>& o) { setResource(o.get()); return *this; }

    bool operator==(const CachedResourceHandle& o) const { return operator==(static_cast<const CachedResourceHandleBase&>(o)); }
    bool operator==(const CachedResourceHandleBase& o) const { return get() == o.get(); }
};

template <class R, class RR> bool operator==(const CachedResourceHandle<R>& h, const RR* res) 
{ 
    return h.get() == res; 
}
template <class R, class RR> bool operator==(const RR* res, const CachedResourceHandle<R>& h) 
{ 
    return h.get() == res; 
}

} // namespace WebCore
