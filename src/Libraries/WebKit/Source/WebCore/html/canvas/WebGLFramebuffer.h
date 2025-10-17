/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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

#include "WebGLObject.h"
#include <variant>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace JSC {
class AbstractSlotVisitor;
}

namespace WTF {
class AbstractLocker;
}

namespace WebCore {

class WebGLRenderbuffer;
class WebGLTexture;

class WebGLFramebuffer final : public WebGLObject {
public:
    virtual ~WebGLFramebuffer();

    static RefPtr<WebGLFramebuffer> create(WebGLRenderingContextBase&);
#if ENABLE(WEBXR)
    static RefPtr<WebGLFramebuffer> createOpaque(WebGLRenderingContextBase&);
#endif

    struct TextureAttachment {
        RefPtr<WebGLTexture> texture;
        GCGLenum texTarget;
        GCGLint level;
        friend bool operator==(const TextureAttachment&, const TextureAttachment&) = default;
    };
    struct TextureLayerAttachment {
        RefPtr<WebGLTexture> texture;
        GCGLint level;
        GCGLint layer;
        friend bool operator==(const TextureLayerAttachment&, const TextureLayerAttachment&) = default;
    };
    using AttachmentEntry = std::variant<RefPtr<WebGLRenderbuffer>, TextureAttachment, TextureLayerAttachment>;

    void setAttachmentForBoundFramebuffer(GCGLenum target, GCGLenum attachment, AttachmentEntry);

    // Below are nonnull. RefPtr instead of Ref due to call site object identity
    // purposes, call site uses i.e pointer operator==.
    using AttachmentObject = std::variant<RefPtr<WebGLRenderbuffer>, RefPtr<WebGLTexture>>;

    // If an object is attached to the currently bound framebuffer, remove it.
    void removeAttachmentFromBoundFramebuffer(const AbstractLocker&, GCGLenum target, AttachmentObject);
    std::optional<AttachmentObject> getAttachmentObject(GCGLenum) const;

    void didBind() { m_hasEverBeenBound = true; }

    // Wrapper for drawBuffersEXT/drawBuffersARB to work around a driver bug.
    void drawBuffers(const Vector<GCGLenum>& bufs);

    GCGLenum getDrawBuffer(GCGLenum);

    void addMembersToOpaqueRoots(const AbstractLocker&, JSC::AbstractSlotVisitor&);

#if ENABLE(WEBXR)
    bool isOpaque() const { return m_isOpaque; }
#endif

    bool isUsable() const { return object() && !isDeleted(); }
    bool isInitialized() const { return m_hasEverBeenBound; }

private:
    enum class Type : bool {
        Plain,
#if ENABLE(WEBXR)
        Opaque
#endif
    };
    WebGLFramebuffer(WebGLRenderingContextBase&, PlatformGLObject, Type);

    void deleteObjectImpl(const AbstractLocker&, GraphicsContextGL*, PlatformGLObject) override;

    // If a given attachment point for the currently bound framebuffer is not null, remove the attached object.
    void removeAttachmentFromBoundFramebuffer(const AbstractLocker&, GCGLenum target, GCGLenum attachment);

    // Check if the framebuffer is currently bound to the given target.
    bool isBound(GCGLenum target) const;

    // Check if a new drawBuffers call should be issued. This is called when we add or remove an attachment.
    void drawBuffersIfNecessary(bool force);

    void setAttachmentInternal(GCGLenum attachment, AttachmentEntry);
    // If a given attachment point for the currently bound framebuffer is not
    // null, remove the attached object.
    void removeAttachmentInternal(const AbstractLocker&, GCGLenum attachment);

    UncheckedKeyHashMap<GCGLenum, AttachmentEntry> m_attachments;
    bool m_hasEverBeenBound { false };
    Vector<GCGLenum> m_drawBuffers;
    Vector<GCGLenum> m_filteredDrawBuffers;
#if ENABLE(WEBXR)
    const bool m_isOpaque;
#endif
};

} // namespace WebCore

#endif
