/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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

#if ENABLE(WEBGL)

#include "GraphicsTypesGL.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class GraphicsContextGL;
class WebCoreOpaqueRoot;
class WebGLRenderingContextBase;

template<typename T, unsigned target = 0>
class WebGLBindingPoint {
    WTF_MAKE_NONCOPYABLE(WebGLBindingPoint);
public:
    WebGLBindingPoint() = default;
    explicit WebGLBindingPoint(RefPtr<T> object)
        : m_object(WTFMove(object))
    {
        if (m_object)
            didBind(*m_object);
    }
    WebGLBindingPoint(WebGLBindingPoint&&) = default;
    ~WebGLBindingPoint() = default;
    WebGLBindingPoint& operator=(WebGLBindingPoint&&) = default;

    WebGLBindingPoint& operator=(RefPtr<T> object)
    {
        if (m_object == object)
            return *this;
        m_object = WTFMove(object);
        if (m_object)
            didBind(*m_object);
        return *this;
    }
    bool operator==(const T* a) const { return a == m_object; }
    bool operator==(const RefPtr<T>& a) const { return a == m_object; }
    explicit operator bool() const { return m_object; }
    T* get() const { return m_object.get(); }
    T* operator->() const { return m_object.get(); }
    T& operator*() const { return *m_object; }
    operator RefPtr<T>() const { return m_object; }

private:
    void didBind(T& object)
    {
        if constexpr(!target)
            object.didBind();
        else
            object.didBind(target);
    }

    RefPtr<T> m_object;
};

class WebGLObject : public RefCounted<WebGLObject> {
public:
    virtual ~WebGLObject();

    WebGLRenderingContextBase* context() const;
    GraphicsContextGL* graphicsContextGL() const;

    PlatformGLObject object() const { return m_object; }

    // deleteObject may not always delete the OpenGL resource.  For programs and
    // shaders, deletion is delayed until they are no longer attached.
    // The AbstractLocker argument enforces at compile time that the objectGraphLock
    // is held. This isn't necessary for all object types, but enough of them that
    // it's done for all of them.
    void deleteObject(const AbstractLocker&, GraphicsContextGL*);

    void onAttached() { ++m_attachmentCount; }
    void onDetached(const AbstractLocker&, GraphicsContextGL*);

    // This indicates whether the client side issue a delete call already, not
    // whether the OpenGL resource is deleted.
    // object()==0 indicates the OpenGL resource is deleted.
    bool isDeleted() const { return m_deleted; }

    // True if this object belongs to the context.
    bool validate(const WebGLRenderingContextBase&) const;

    Lock& objectGraphLockForContext();

protected:
    WebGLObject(WebGLRenderingContextBase&, PlatformGLObject);

    void runDestructor();

    // deleteObjectImpl should be only called once to delete the OpenGL resource.
    virtual void deleteObjectImpl(const AbstractLocker&, GraphicsContextGL*, PlatformGLObject) = 0;

    WeakPtr<WebGLRenderingContextBase> m_context;
private:
    PlatformGLObject m_object { 0 };
    unsigned m_attachmentCount { 0 };
    bool m_deleted { false };
};

template<typename T>
PlatformGLObject objectOrZero(const T& object)
{
    return object ? object->object() : 0;
}

WebCoreOpaqueRoot root(WebGLObject*);

} // namespace WebCore

#endif
