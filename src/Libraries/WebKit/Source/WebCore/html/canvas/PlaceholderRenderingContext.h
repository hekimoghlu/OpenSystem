/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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

#if ENABLE(OFFSCREEN_CANVAS)

#include "CanvasRenderingContext.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class PlaceholderRenderingContext;

// Thread-safe interface to submit frames from worker to the placeholder rendering context.
class PlaceholderRenderingContextSource : public ThreadSafeRefCounted<PlaceholderRenderingContextSource> {
    WTF_MAKE_TZONE_ALLOCATED(PlaceholderRenderingContextSource);
    WTF_MAKE_NONCOPYABLE(PlaceholderRenderingContextSource);
public:
    static Ref<PlaceholderRenderingContextSource> create(PlaceholderRenderingContext&);
    virtual ~PlaceholderRenderingContextSource() = default;

    // Called by the offscreen context to submit the frame.
    void setPlaceholderBuffer(ImageBuffer&);

    // Called by the placeholder context to attach to compositor layer.
    void setContentsToLayer(GraphicsLayer&);

private:
    explicit PlaceholderRenderingContextSource(PlaceholderRenderingContext&);

    WeakPtr<PlaceholderRenderingContext> m_placeholder; // For main thread use.
    Lock m_lock;
    RefPtr<GraphicsLayerAsyncContentsDisplayDelegate> m_delegate WTF_GUARDED_BY_LOCK(m_lock);
};

class PlaceholderRenderingContext final : public CanvasRenderingContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PlaceholderRenderingContext);
public:
    static std::unique_ptr<PlaceholderRenderingContext> create(HTMLCanvasElement&);

    HTMLCanvasElement& canvas() const;
    IntSize size() const;
    void setPlaceholderBuffer(Ref<ImageBuffer>&&);

    Ref<PlaceholderRenderingContextSource> source() const { return m_source; }

private:
    PlaceholderRenderingContext(HTMLCanvasElement&);
    void setContentsToLayer(GraphicsLayer&) final;

    Ref<PlaceholderRenderingContextSource> m_source;
};

}

SPECIALIZE_TYPE_TRAITS_CANVASRENDERINGCONTEXT(WebCore::PlaceholderRenderingContext, isPlaceholder())

#endif
