/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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

#if ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)

#include <WebCore/ScrollingCoordinatorMac.h>

namespace WebKit {

class WebPage;

class TiledCoreAnimationScrollingCoordinator final : public WebCore::ScrollingCoordinatorMac {
public:
    static Ref<TiledCoreAnimationScrollingCoordinator> create(WebPage* page)
    {
        return adoptRef(*new TiledCoreAnimationScrollingCoordinator(page));
    }

private:
    explicit TiledCoreAnimationScrollingCoordinator(WebPage*);
    ~TiledCoreAnimationScrollingCoordinator();

    void pageDestroyed() final;
    void hasNodeWithAnimatedScrollChanged(bool) final;
    
    WeakPtr<WebPage> m_page;
};

} // namespace WebKit

#endif // ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)
