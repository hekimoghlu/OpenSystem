/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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

#include "GraphicsContextGL.h"
#include "GraphicsTypesGL.h"
#include <optional>
#include <wtf/Noncopyable.h>

namespace WebCore {

class ScopedRestoreTextureBinding {
    WTF_MAKE_NONCOPYABLE(ScopedRestoreTextureBinding);

public:
    ScopedRestoreTextureBinding(GCGLenum bindingPointQuery, GCGLenum bindingPoint, bool condition = true);
    ~ScopedRestoreTextureBinding();

private:
    GCGLenum m_bindingPoint { 0 };
    GCGLuint m_bindingValue { 0u };
};

class ScopedBufferBinding {
    WTF_MAKE_NONCOPYABLE(ScopedBufferBinding);

public:
    ScopedBufferBinding(GCGLenum bindingPoint, GCGLuint bindingValue, bool condition = true);
    ~ScopedBufferBinding();

private:
    static constexpr GCGLenum query(GCGLenum bindingPoint)
    {
        if (bindingPoint == GraphicsContextGL::PIXEL_PACK_BUFFER)
            return GraphicsContextGL::PIXEL_PACK_BUFFER_BINDING;
        ASSERT(bindingPoint == GraphicsContextGL::PIXEL_UNPACK_BUFFER);
        return GraphicsContextGL::PIXEL_UNPACK_BUFFER_BINDING;
    }
    GCGLint m_bindingPoint { 0 };
    GCGLuint m_bindingValue { 0u };
};

class ScopedRestoreReadFramebufferBinding {
    WTF_MAKE_NONCOPYABLE(ScopedRestoreReadFramebufferBinding);

public:
    ScopedRestoreReadFramebufferBinding(bool isForWebGL2, GCGLuint restoreValue)
        : m_framebufferTarget(isForWebGL2 ? GraphicsContextGL::READ_FRAMEBUFFER : GraphicsContextGL::FRAMEBUFFER)
        , m_bindingValue(restoreValue)
    {
    }
    ScopedRestoreReadFramebufferBinding(bool isForWebGL2, GCGLuint restoreValue, GCGLuint value)
        : m_framebufferTarget(isForWebGL2 ? GraphicsContextGL::READ_FRAMEBUFFER : GraphicsContextGL::FRAMEBUFFER)
        , m_bindingValue(restoreValue)
    {
        bindFramebuffer(value);
    }
    ~ScopedRestoreReadFramebufferBinding();
    void markBindingChanged()
    {
        m_bindingChanged = true;
    }
    void bindFramebuffer(GCGLuint bindingValue);
    GCGLuint framebufferTarget() const { return m_framebufferTarget; }

private:
    const GCGLenum m_framebufferTarget;
    GCGLuint m_bindingValue { 0u };
    bool m_bindingChanged { false };
};

class ScopedPixelStorageMode {
    WTF_MAKE_NONCOPYABLE(ScopedPixelStorageMode);

public:
    explicit ScopedPixelStorageMode(GCGLenum name, bool condition = true);
    ScopedPixelStorageMode(GCGLenum name, GCGLint value, bool condition = true);
    ~ScopedPixelStorageMode();
    void pixelStore(GCGLint value);
    operator GCGLint() const
    {
        ASSERT(m_name && !m_valueChanged);
        return m_originalValue;
    }

private:
    const GCGLenum m_name;
    bool m_valueChanged { false };
    GCGLint m_originalValue { 0 };
};

class ScopedTexture {
    WTF_MAKE_NONCOPYABLE(ScopedTexture);

public:
    ScopedTexture();
    ~ScopedTexture();
    operator GCGLuint() const
    {
        return m_object;
    }

private:
    GCGLuint m_object { 0u };
};

class ScopedFramebuffer {
    WTF_MAKE_NONCOPYABLE(ScopedFramebuffer);

public:
    ScopedFramebuffer();
    ~ScopedFramebuffer();
    operator GCGLuint() const
    {
        return m_object;
    }
private:
    GCGLuint m_object { 0 };
};

class ScopedGLFence {
    WTF_MAKE_NONCOPYABLE(ScopedGLFence);

public:
    ScopedGLFence() = default;
    ScopedGLFence(ScopedGLFence&& other)
        : m_object(std::exchange(other.m_object, { }))
    {
    }
    ~ScopedGLFence() { reset(); }
    ScopedGLFence& operator=(ScopedGLFence&& other)
    {
        if (this != &other) {
            reset();
            m_object = std::exchange(other.m_object, { });
        }
        return *this;
    }
    void reset();
    void abandon() { m_object = { }; }
    void fenceSync();
    GCGLsync get() const { return m_object; }
    operator GCGLsync() const { return m_object; }
    operator bool() const { return m_object; }

private:
    GCGLsync m_object { };
};

class ScopedGLCapability {
    WTF_MAKE_NONCOPYABLE(ScopedGLCapability);
public:
    ScopedGLCapability(GCGLenum capability, bool enable);
    ~ScopedGLCapability();

private:
    const GCGLenum m_capability;
    const std::optional<bool> m_original;
};

bool platformIsANGLEAvailable();

}

#endif
