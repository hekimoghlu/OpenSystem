/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#include "config.h"
#include "WebGLFramebuffer.h"

#if ENABLE(WEBGL)

#include "WebCoreOpaqueRootInlines.h"
#include "WebGLDrawBuffers.h"
#include "WebGLRenderingContextBase.h"
#include <JavaScriptCore/SlotVisitor.h>
#include <JavaScriptCore/SlotVisitorInlines.h>
#include <wtf/Lock.h>
#include <wtf/Locker.h>

namespace WebCore {

static void entryAddMembersToOpaqueRoots(const WebGLFramebuffer::AttachmentEntry& entry, const AbstractLocker&, JSC::AbstractSlotVisitor& visitor)
{
    // This runs on GC thread, so use const RefPtr<>& to make ensure no RefPtrs are created.
    switchOn(entry,
        [&](const RefPtr<WebGLRenderbuffer>& renderbuffer) {
            addWebCoreOpaqueRoot(visitor, renderbuffer.get());
        },
        [&](const WebGLFramebuffer::TextureAttachment& textureAttachment) {
            addWebCoreOpaqueRoot(visitor, textureAttachment.texture.get());
        },
        [&](const WebGLFramebuffer::TextureLayerAttachment& layerAttachment) {
            addWebCoreOpaqueRoot(visitor, layerAttachment.texture.get());
        });
}

static void entryDetachAndClear(WebGLFramebuffer::AttachmentEntry& entry, const AbstractLocker& locker, GraphicsContextGL* gl)
{
    switchOn(entry,
        [&](RefPtr<WebGLRenderbuffer>& renderbuffer) {
            renderbuffer->onDetached(locker, gl);
            renderbuffer = nullptr;
        },
        [&](WebGLFramebuffer::TextureAttachment& textureAttachment) {
            textureAttachment.texture->onDetached(locker, gl);
            textureAttachment.texture = nullptr;
        },
        [&](WebGLFramebuffer::TextureLayerAttachment& layerAttachment) {
            layerAttachment.texture->onDetached(locker, gl);
            layerAttachment.texture = nullptr;
        });
}

static void entryAttach(WebGLFramebuffer::AttachmentEntry& entry)
{
    switchOn(entry,
        [&](RefPtr<WebGLRenderbuffer>& renderbuffer) {
            renderbuffer->onAttached();
        },
        [&](WebGLFramebuffer::TextureAttachment& textureAttachment) {
            textureAttachment.texture->onAttached();
        },
        [&](WebGLFramebuffer::TextureLayerAttachment& layerAttachment) {
            layerAttachment.texture->onAttached();
        });
}

static void entryContextSetAttachment(const WebGLFramebuffer::AttachmentEntry& entry, GraphicsContextGL* gl, GCGLenum target, GCGLenum attachment)
{
    switchOn(entry,
        [&] (RefPtr<WebGLRenderbuffer> renderbuffer) {
            gl->framebufferRenderbuffer(target, attachment, GraphicsContextGL::RENDERBUFFER, objectOrZero(renderbuffer));
        },
        [&] (WebGLFramebuffer::TextureAttachment textureAttachment) {
            gl->framebufferTexture2D(target, attachment, textureAttachment.texTarget, objectOrZero(textureAttachment.texture), textureAttachment.level);
        },
        [&] (WebGLFramebuffer::TextureLayerAttachment layerAttachment) {
            gl->framebufferTextureLayer(target, attachment, objectOrZero(layerAttachment.texture), layerAttachment.level, layerAttachment.layer);
        });
}

static WebGLFramebuffer::AttachmentObject entryObject(const WebGLFramebuffer::AttachmentEntry& entry)
{
    return switchOn(entry,
        [&] (RefPtr<WebGLRenderbuffer> renderbuffer) -> WebGLFramebuffer::AttachmentObject {
            return renderbuffer;
        },
        [&] (const WebGLFramebuffer::TextureAttachment& textureAttachment) -> WebGLFramebuffer::AttachmentObject {
            return textureAttachment.texture;
        },
        [&] (const WebGLFramebuffer::TextureLayerAttachment& layerAttachment) -> WebGLFramebuffer::AttachmentObject {
            return layerAttachment.texture;
        });
}

static bool entryHasObject(const WebGLFramebuffer::AttachmentEntry& entry)
{
    return switchOn(entryObject(entry),
        [&] (auto&& object) -> bool {
            return object;
        });
}

RefPtr<WebGLFramebuffer> WebGLFramebuffer::create(WebGLRenderingContextBase& context)
{
    auto object = context.protectedGraphicsContextGL()->createFramebuffer();
    if (!object)
        return nullptr;
    return adoptRef(*new WebGLFramebuffer { context, object, Type::Plain });
}

#if ENABLE(WEBXR)
RefPtr<WebGLFramebuffer> WebGLFramebuffer::createOpaque(WebGLRenderingContextBase& context)
{
    auto object = context.protectedGraphicsContextGL()->createFramebuffer();
    if (!object)
        return nullptr;
    return adoptRef(*new WebGLFramebuffer { context, object, Type::Opaque });
}
#endif

WebGLFramebuffer::WebGLFramebuffer(WebGLRenderingContextBase& context, PlatformGLObject object, Type type)
    : WebGLObject(context, object)
#if ENABLE(WEBXR)
    , m_isOpaque(type == Type::Opaque)
#endif
{
    UNUSED_PARAM(type);
}

WebGLFramebuffer::~WebGLFramebuffer()
{
    if (!context())
        return;

    runDestructor();
}

void WebGLFramebuffer::setAttachmentForBoundFramebuffer(GCGLenum target, GCGLenum attachment, AttachmentEntry entry)
{
    ASSERT(object());
    ASSERT(isBound(target));
    auto attachmentCount = m_attachments.size();
    RefPtr gl = context()->graphicsContextGL();
    if (attachment == GraphicsContextGL::DEPTH_STENCIL_ATTACHMENT && context()->isWebGL2()) {
        setAttachmentInternal(GraphicsContextGL::STENCIL_ATTACHMENT, entry);
        entryContextSetAttachment(entry, gl.get(), target, GraphicsContextGL::STENCIL_ATTACHMENT);
        attachment = GraphicsContextGL::DEPTH_ATTACHMENT;
    }
    setAttachmentInternal(attachment, entry);
    entryContextSetAttachment(entry, gl.get(), target, attachment);

    if (attachmentCount != m_attachments.size())
        drawBuffersIfNecessary(false);
}

std::optional<WebGLFramebuffer::AttachmentObject> WebGLFramebuffer::getAttachmentObject(GCGLenum attachment) const
{
    if (!object())
        return std::nullopt;
    auto it = m_attachments.find(attachment);
    if (it == m_attachments.end())
        return std::nullopt;
    return entryObject(it->value);
}

void WebGLFramebuffer::removeAttachmentFromBoundFramebuffer(const AbstractLocker& locker, GCGLenum target, AttachmentObject removedObject)
{
    ASSERT(isBound(target));
    if (!object())
        return;
    auto attachmentCount = m_attachments.size();
    bool checkMore = true;
    RefPtr gl = context()->graphicsContextGL();
    do {
        checkMore = false;
        for (auto it = m_attachments.begin(); it != m_attachments.end(); ++it) {
            if (entryObject(it->value) != removedObject)
                continue;
            GCGLenum attachment = it->key;
            auto entry = WTFMove(it->value);
            m_attachments.remove(it);
            checkMore = true;
            entryDetachAndClear(entry, locker, gl.get());
            entryContextSetAttachment(entry, gl.get(), target, attachment);
            break;
        }
    } while (checkMore);
    if (attachmentCount != m_attachments.size())
        drawBuffersIfNecessary(false);
}

void WebGLFramebuffer::deleteObjectImpl(const AbstractLocker& locker, GraphicsContextGL* context3d, PlatformGLObject object)
{
    for (auto& entry : m_attachments.values())
        entryDetachAndClear(entry, locker, context3d);

    context3d->deleteFramebuffer(object);
}

bool WebGLFramebuffer::isBound(GCGLenum target) const
{
    return (context()->getFramebufferBinding(target) == this);
}

void WebGLFramebuffer::drawBuffers(const Vector<GCGLenum>& bufs)
{
    m_drawBuffers = bufs;
    m_filteredDrawBuffers.resize(m_drawBuffers.size());
    for (auto& buffer : m_filteredDrawBuffers)
        buffer = GraphicsContextGL::NONE;
    drawBuffersIfNecessary(true);
}

void WebGLFramebuffer::drawBuffersIfNecessary(bool force)
{
    if (context()->isWebGL2() || context()->m_webglDrawBuffers) {
        bool reset = force;
        // This filtering works around graphics driver bugs on macOS.
        for (size_t i = 0; i < m_drawBuffers.size(); ++i) {
            if (m_drawBuffers[i] != GraphicsContextGL::NONE && m_attachments.contains(m_drawBuffers[i])) {
                if (m_filteredDrawBuffers[i] != m_drawBuffers[i]) {
                    m_filteredDrawBuffers[i] = m_drawBuffers[i];
                    reset = true;
                }
            } else {
                if (m_filteredDrawBuffers[i] != GraphicsContextGL::NONE) {
                    m_filteredDrawBuffers[i] = GraphicsContextGL::NONE;
                    reset = true;
                }
            }
        }
        if (reset) {
            if (context()->isWebGL2())
                context()->protectedGraphicsContextGL()->drawBuffers(m_filteredDrawBuffers);
            else
                context()->protectedGraphicsContextGL()->drawBuffersEXT(m_filteredDrawBuffers);
        }
    }
}

GCGLenum WebGLFramebuffer::getDrawBuffer(GCGLenum drawBuffer)
{
    int index = static_cast<int>(drawBuffer - GraphicsContextGL::DRAW_BUFFER0_EXT);
    ASSERT(index >= 0);
    if (index < static_cast<int>(m_drawBuffers.size()))
        return m_drawBuffers[index];
    if (drawBuffer == GraphicsContextGL::DRAW_BUFFER0_EXT)
        return GraphicsContextGL::COLOR_ATTACHMENT0;
    return GraphicsContextGL::NONE;
}

void WebGLFramebuffer::addMembersToOpaqueRoots(const AbstractLocker& locker, JSC::AbstractSlotVisitor& visitor)
{
    for (auto& entry : m_attachments)
        entryAddMembersToOpaqueRoots(entry.value, locker, visitor);
}

void WebGLFramebuffer::setAttachmentInternal(GCGLenum attachment, AttachmentEntry entry)
{
    if (!context()) {
        // Context has been deleted - should not be calling this.
        return;
    }
    Locker locker { objectGraphLockForContext() };

    auto it = m_attachments.find(attachment);
    if (it != m_attachments.end()) {
        if (entry == it->value)
            return;
        entryDetachAndClear(it->value, locker, context()->protectedGraphicsContextGL().get());
        m_attachments.remove(it);
    }
    if (!entryHasObject(entry))
        return;
    auto result = m_attachments.add(attachment, WTFMove(entry));
    entryAttach(result.iterator->value);
}

}

#endif // ENABLE(WEBGL)
