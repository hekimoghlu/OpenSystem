/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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

#include "RendererBufferFormat.h"
#include <wtf/AbstractRefCounted.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

typedef struct _cairo cairo_t;

#if USE(GTK4)
typedef struct _GdkSnapshot GdkSnapshot;
typedef GdkSnapshot GtkSnapshot;
#endif

namespace WebCore {
class IntRect;
class NativeImage;
}

namespace WebKit {

class LayerTreeContext;
class WebPageProxy;

class AcceleratedBackingStore : public AbstractRefCounted {
    WTF_MAKE_TZONE_ALLOCATED(AcceleratedBackingStore);
    WTF_MAKE_NONCOPYABLE(AcceleratedBackingStore);
public:
    static bool checkRequirements();
    static RefPtr<AcceleratedBackingStore> create(WebPageProxy&);
    virtual ~AcceleratedBackingStore() = default;

    virtual void update(const LayerTreeContext&) { }
#if USE(GTK4)
    virtual bool snapshot(GtkSnapshot*) = 0;
#else
    virtual bool paint(cairo_t*, const WebCore::IntRect&) = 0;
#endif
    virtual void realize() { };
    virtual void unrealize() { };
    virtual int renderHostFileDescriptor() { return -1; }
    virtual RendererBufferFormat bufferFormat() const { return { }; }
    virtual RefPtr<WebCore::NativeImage> bufferAsNativeImageForTesting() const = 0;

protected:
    explicit AcceleratedBackingStore(WebPageProxy&);

    WeakPtr<WebPageProxy> m_webPage;
};

} // namespace WebKit
