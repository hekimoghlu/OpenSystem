/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionContext.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "WKWebExtensionControllerInternal.h"
#import "WKWebViewInternal.h"
#import "WebExtensionController.h"
#import "WebProcessProxy.h"
#import <wtf/EnumTraits.h>

namespace WebKit {

void WebExtensionContext::addListener(WebCore::FrameIdentifier frameIdentifier, WebExtensionEventListenerType listenerType, WebExtensionContentWorldType contentWorldType)
{
    RefPtr frame = WebFrameProxy::webFrame(frameIdentifier);
    if (!frame)
        return;

    RELEASE_LOG_DEBUG(Extensions, "Registered event listener for type %{public}hhu in %{public}@ world", enumToUnderlyingType(listenerType), (NSString *)toDebugString(contentWorldType));

    if (!protectedExtension()->backgroundContentIsPersistent() && isBackgroundPage(frameIdentifier))
        m_backgroundContentEventListeners.add(listenerType);

    auto result = m_eventListenerFrames.add({ listenerType, contentWorldType }, WeakFrameCountedSet { });
    result.iterator->value.add(*frame);
}

void WebExtensionContext::removeListener(WebCore::FrameIdentifier frameIdentifier, WebExtensionEventListenerType listenerType, WebExtensionContentWorldType contentWorldType, size_t removedCount)
{
    ASSERT(removedCount);

    RefPtr frame = WebFrameProxy::webFrame(frameIdentifier);
    if (!frame)
        return;

    RELEASE_LOG_DEBUG(Extensions, "Unregistered %{public}zu event listener(s) for type %{public}hhu in %{public}@ world", removedCount, enumToUnderlyingType(listenerType), (NSString *)toDebugString(contentWorldType));

    if (!protectedExtension()->backgroundContentIsPersistent() && isBackgroundPage(frameIdentifier)) {
        for (size_t i = 0; i < removedCount; ++i)
            m_backgroundContentEventListeners.remove(listenerType);
    }

    auto iterator = m_eventListenerFrames.find({ listenerType, contentWorldType });
    if (iterator == m_eventListenerFrames.end())
        return;

    for (size_t i = 0; i < removedCount; ++i)
        iterator->value.remove(*frame);

    if (!iterator->value.isEmptyIgnoringNullReferences())
        return;

    m_eventListenerFrames.remove(iterator);
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
