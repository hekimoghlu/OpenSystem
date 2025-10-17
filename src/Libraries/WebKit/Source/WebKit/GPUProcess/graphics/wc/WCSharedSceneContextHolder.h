/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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

#if USE(GRAPHICS_LAYER_WC)

#include "WCSceneContext.h"
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>

namespace WebKit {

// Creating a WCSceneContext for a window fails if the window already
// has one. While navigating to a new page with a new WebProcess, the
// provisional WebKit::WebPage in the new WebProcess has the same
// window handle. WCSceneContext should be shared among all
// RemoteWCLayerTreeHosts that render to the same native window.
class WCSharedSceneContextHolder {
    WTF_MAKE_NONCOPYABLE(WCSharedSceneContextHolder);
public:
    WCSharedSceneContextHolder() = default;

    struct Holder : RefCounted<Holder> {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        int64_t nativeWindow;
        // The following member is accessed in the OpenGL thread
        std::optional<WCSceneContext> context;
    };

    Ref<Holder> ensureHolderForWindow(int64_t nativeWindow)
    {
        ASSERT(RunLoop::isMain());
        auto result = m_hash.add(nativeWindow, nullptr);
        if (!result.isNewEntry) 
            return *result.iterator->value;
        auto holder = adoptRef(*new Holder);
        holder->nativeWindow = nativeWindow;
        result.iterator->value = holder.ptr();
        return holder;
    }

    RefPtr<Holder> removeHolder(Ref<Holder>&& holder)
    {
        ASSERT(RunLoop::isMain());
        if (!holder->hasOneRef()) 
            return nullptr;
        m_hash.remove(holder->nativeWindow);
        return holder;
    }

private:
    HashMap<int64_t, Holder*> m_hash;
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
