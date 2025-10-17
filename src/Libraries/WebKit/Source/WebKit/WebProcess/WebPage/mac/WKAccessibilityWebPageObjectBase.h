/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#import <WebCore/FloatPoint.h>
#import <WebCore/FrameIdentifier.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/PageIdentifier.h>
#import <wtf/Lock.h>
#import <wtf/NakedPtr.h>
#import <wtf/WeakObjCPtr.h>

namespace WebKit {
class WebPage;
}

namespace WebCore {
class AXCoreObject;
}

@interface WKAccessibilityWebPageObjectBase : NSObject {
    NakedPtr<WebKit::WebPage> m_page;
    Markable<WebCore::PageIdentifier> m_pageID;
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    Lock m_cacheLock;
    WebCore::FloatPoint m_position WTF_GUARDED_BY_LOCK(m_cacheLock);
    WebCore::IntSize m_size WTF_GUARDED_BY_LOCK(m_cacheLock);
    ThreadSafeWeakPtr<WebCore::AXCoreObject> m_isolatedTreeRoot;

    Lock m_windowLock;
    WeakObjCPtr<id> m_window;
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)

    WebCore::IntPoint m_remoteFrameOffset;
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    Lock m_parentLock;
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    RetainPtr<id> m_parent;
    bool m_hasMainFramePlugin;
    std::optional<WebCore::FrameIdentifier> m_frameID;
}

- (void)setWebPage:(NakedPtr<WebKit::WebPage>)page;
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
- (void)setPosition:(const WebCore::FloatPoint&)point;
- (void)setSize:(const WebCore::IntSize&)size;
- (void)setIsolatedTreeRoot:(NakedPtr<WebCore::AXCoreObject>)root;
- (void)setWindow:(id)window;
#endif
- (void)setRemoteParent:(id)parent;
- (void)setRemoteFrameOffset:(WebCore::IntPoint)offset;
- (void)setHasMainFramePlugin:(bool)hasPlugin;
- (void)setFrameIdentifier:(const WebCore::FrameIdentifier&)frameID;

- (id)accessibilityRootObjectWrapper;
- (id)accessibilityFocusedUIElement;
- (WebCore::IntPoint)accessibilityRemoteFrameOffset;
- (WebCore::LocalFrame *)remoteLocalFrame;

@end
