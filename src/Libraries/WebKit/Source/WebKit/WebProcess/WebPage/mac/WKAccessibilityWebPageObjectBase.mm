/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
#import "config.h"
#import "WKAccessibilityWebPageObjectBase.h"

#import "WebFrame.h"
#import "WebPage.h"
#import "WKArray.h"
#import "WKNumber.h"
#import "WKRetainPtr.h"
#import "WKSharedAPICast.h"
#import "WKString.h"
#import "WKStringCF.h"
#import <WebCore/AXObjectCache.h>
#import <WebCore/Document.h>
#import <WebCore/FrameTree.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/Page.h>
#import <WebCore/RemoteFrame.h>
#import <WebCore/ScrollView.h>
#import <WebCore/Scrollbar.h>

namespace ax = WebCore::Accessibility;

@implementation WKAccessibilityWebPageObjectBase

- (NakedPtr<WebCore::AXObjectCache>)axObjectCache
{
    ASSERT(isMainRunLoop());

    if (!m_page)
        return nullptr;

    auto page = m_page->corePage();
    if (!page)
        return nullptr;

    if (auto* localMainFrame = dynamicDowncast<WebCore::LocalFrame>(page->mainFrame())) {
        if (auto* document = localMainFrame->document())
            return document->axObjectCache();
    } else if (RefPtr remoteLocalFrame = [self remoteLocalFrame]) {
        CheckedPtr document = remoteLocalFrame ? remoteLocalFrame->document() : nullptr;
        return document ? document->axObjectCache() : nullptr;
    }

    return nullptr;
}

- (void)enableAccessibilityForAllProcesses
{
    // Immediately enable accessibility in the current web process, otherwise this
    // will happen asynchronously and could break certain flows (e.g., attribute
    // requests).
    if (!WebCore::AXObjectCache::accessibilityEnabled())
        WebCore::AXObjectCache::enableAccessibility();

    if (m_page)
        m_page->enableAccessibilityForAllProcesses();
}

- (id)accessibilityPluginObject
{
    ASSERT(isMainRunLoop());
    auto retrieveBlock = [&self]() -> id {
        id axPlugin = nil;
        callOnMainRunLoopAndWait([&axPlugin, &self] {
            if (self->m_page)
                axPlugin = self->m_page->accessibilityObjectForMainFramePlugin();
        });
        return axPlugin;
    };
    
    return retrieveBlock();
}

- (id)accessibilityRootObjectWrapper
{
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    if (!isMainRunLoop()) {
        if (RefPtr root = m_isolatedTreeRoot.get())
            return root->wrapper();
    }
#endif

    return ax::retrieveAutoreleasedValueFromMainThread<id>([protectedSelf = retainPtr(self)] () -> RetainPtr<id> {
        if (!WebCore::AXObjectCache::accessibilityEnabled())
            [protectedSelf enableAccessibilityForAllProcesses];

        if (protectedSelf.get()->m_hasMainFramePlugin) {
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
            // Even though we want to serve the PDF plugin tree for main-frame plugins, we still need to make sure the isolated tree
            // is built, so that when text annotations are created on-the-fly as users focus on text fields,
            // isolated objects are able to be attached to those text annotation object wrappers.
            // If they aren't, we never have a backing object to serve any requests from.
            if (auto cache = protectedSelf.get().axObjectCache)
                cache->buildAccessibilityTreeIfNeeded();
#endif
            return protectedSelf.get().accessibilityPluginObject;
        }

        if (auto cache = protectedSelf.get().axObjectCache) {
            if (auto* root = cache->rootObject())
                return root->wrapper();
        }

        return nil;
    });
}

- (void)setWebPage:(NakedPtr<WebKit::WebPage>)page
{
    ASSERT(isMainRunLoop());

    m_page = page;

    if (page) {
        m_pageID = page->identifier();
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
        [self setPosition:page->accessibilityPosition()];
        [self setSize:page->size()];
#endif
        auto* frame = dynamicDowncast<WebCore::LocalFrame>(page->mainFrame());
        m_hasMainFramePlugin = frame && frame->document() ? frame->document()->isPluginDocument() : false;
    } else {
        m_pageID = std::nullopt;
        m_hasMainFramePlugin = false;
    }
}

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
- (void)setPosition:(const WebCore::FloatPoint&)point
{
    ASSERT(isMainRunLoop());
    Locker locker { m_cacheLock };
    m_position = point;
}

- (void)setSize:(const WebCore::IntSize&)size
{
    ASSERT(isMainRunLoop());
    Locker locker { m_cacheLock };
    m_size = size;
}

- (void)setIsolatedTreeRoot:(NakedPtr<WebCore::AXCoreObject>)root
{
    ASSERT(isMainRunLoop());

    if (m_hasMainFramePlugin) {
        // Do not set the isolated tree root for main-frame plugins, as that would prevent serving the root
        // of the plugin accessiblity tree.
        return;
    }
    m_isolatedTreeRoot = root.get();
}

- (void)setWindow:(id)window
{
    ASSERT(isMainRunLoop());
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    Locker lock { m_windowLock };
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    m_window = window;
}
#endif

- (void)setHasMainFramePlugin:(bool)hasPlugin
{
    ASSERT(isMainRunLoop());
    m_hasMainFramePlugin = hasPlugin;
}

- (void)setRemoteFrameOffset:(WebCore::IntPoint)offset
{
    ASSERT(isMainRunLoop());
    m_remoteFrameOffset = offset;
}

- (WebCore::IntPoint)accessibilityRemoteFrameOffset
{
    return m_remoteFrameOffset;
}

- (void)setRemoteParent:(id)parent
{
    ASSERT(isMainRunLoop());

#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    Locker lock { m_parentLock };
#endif // ENABLE(ACCESSIBILITY_ISOLATED_TREE)
    m_parent = parent;
}

- (void)setFrameIdentifier:(const WebCore::FrameIdentifier&)frameID
{
    m_frameID = frameID;
}

- (id)accessibilityFocusedUIElement
{
    return [[self accessibilityRootObjectWrapper] accessibilityFocusedUIElement];
}

- (WebCore::LocalFrame *)remoteLocalFrame
{
    if (!m_page)
        return nullptr;

    RefPtr page = m_page->corePage();
    if (!page)
        return nullptr;

    for (auto& rootFrame : page->rootFrames()) {
        if (rootFrame->frameID() == m_frameID)
            return rootFrame.ptr();
    }

    return nullptr;
}

@end
