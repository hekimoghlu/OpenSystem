/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#if ENABLE(WEBXR) && PLATFORM(IOS)

#import "config.h"
#import "PlatformXRSystem.h"

#import "PlatformXRARKit.h"
#import <wtf/NeverDestroyed.h>

namespace WebKit {

PlatformXRCoordinator* PlatformXRSystem::xrCoordinator()
{
#if USE(ARKITXR_IOS)
    static LazyNeverDestroyed<ARKitCoordinator> xrCoordinator;
    static std::once_flag once;
    std::call_once(once, [] {
        xrCoordinator.construct();
    });
    return &xrCoordinator.get();
#else
    return nullptr;
#endif // USE(ARKITXR_IOS)
}

} // namespace WebKit

#endif // ENABLE(WEBXR) && PLATFORM(IOS)
