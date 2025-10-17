/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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

#include <WebCore/Damage.h>
#include <WebCore/IntSize.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WTF {
class RunLoop;
}

namespace WebCore {
class Region;
}

namespace WebKit {

class ThreadedCompositor;
class WebPage;

class AcceleratedSurface {
    WTF_MAKE_NONCOPYABLE(AcceleratedSurface);
    WTF_MAKE_TZONE_ALLOCATED(AcceleratedSurface);
public:
    static std::unique_ptr<AcceleratedSurface> create(ThreadedCompositor&, WebPage&, Function<void()>&& frameCompleteHandler);
    virtual ~AcceleratedSurface() = default;

    virtual uint64_t window() const { ASSERT_NOT_REACHED(); return 0; }
    virtual uint64_t surfaceID() const { ASSERT_NOT_REACHED(); return 0; }
    virtual bool resize(const WebCore::IntSize&);
    virtual bool shouldPaintMirrored() const { return false; }

    virtual void didCreateGLContext() { }
    virtual void willDestroyGLContext() { }
    virtual void finalize() { }
    virtual void willRenderFrame() { }
    virtual void didRenderFrame() { }

#if ENABLE(DAMAGE_TRACKING)
    virtual const WebCore::Damage& addDamage(const WebCore::Damage&) { return WebCore::Damage::invalid(); };
#endif

    virtual void didCreateCompositingRunLoop(WTF::RunLoop&) { }
    virtual void willDestroyCompositingRunLoop() { }

#if PLATFORM(WPE) && USE(GBM) && ENABLE(WPE_PLATFORM)
    virtual void preferredBufferFormatsDidChange() { }
#endif

    virtual void visibilityDidChange(bool) { }
    virtual bool backgroundColorDidChange();

    void clearIfNeeded();

protected:
    AcceleratedSurface(WebPage&, Function<void()>&& frameCompleteHandler);

    void frameComplete() const;

    WeakRef<WebPage> m_webPage;
    Function<void()> m_frameCompleteHandler;
    WebCore::IntSize m_size;
    std::atomic<bool> m_isOpaque { true };
};

} // namespace WebKit
