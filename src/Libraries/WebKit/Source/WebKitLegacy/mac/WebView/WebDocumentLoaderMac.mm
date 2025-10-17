/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#import "WebDocumentLoaderMac.h"

#import "WebKitVersionChecks.h"
#import "WebView.h"
#import <WebCore/FrameDestructionObserverInlines.h>

using namespace WebCore;

WebDocumentLoaderMac::WebDocumentLoaderMac(const ResourceRequest& request, const SubstituteData& substituteData)
    : DocumentLoader(request, substituteData)
    , m_dataSource(nil)
    , m_isDataSourceRetained(false)
{
}

static inline bool needsDataLoadWorkaround(WebView *webView)
{
#if !PLATFORM(IOS_FAMILY)
    static bool needsWorkaround = !WebKitLinkedOnOrAfter(WEBKIT_FIRST_VERSION_WITHOUT_ADOBE_INSTALLER_QUIRK) 
                                  && [[[NSBundle mainBundle] bundleIdentifier] isEqualToString:@"com.adobe.Installers.Setup"];
    return needsWorkaround;
#else
    return NO;
#endif
}

void WebDocumentLoaderMac::setDataSource(WebDataSource *dataSource, WebView *webView)
{
    ASSERT(!m_dataSource);
    ASSERT(!m_isDataSourceRetained);

    m_dataSource = dataSource;
    retainDataSource();

    m_resourceLoadDelegate = [webView resourceLoadDelegate];
    m_downloadDelegate = [webView downloadDelegate];
    
    // Some clients run the run loop in a way that prevents the data load timer
    // from firing. We work around that issue here. See <rdar://problem/5266289>
    // and <rdar://problem/5049509>.
    if (needsDataLoadWorkaround(webView))
        setDeferMainResourceDataLoad(false);
}

WebDataSource *WebDocumentLoaderMac::dataSource() const
{
    return m_dataSource;
}

void WebDocumentLoaderMac::attachToFrame()
{
    DocumentLoader::attachToFrame();

    retainDataSource();
}

void WebDocumentLoaderMac::detachFromFrame(LoadWillContinueInAnotherProcess loadWillContinueInAnotherProcess)
{
    DocumentLoader::detachFromFrame(loadWillContinueInAnotherProcess);

    if (m_loadingResources.isEmpty())
        releaseDataSource();

    // FIXME: What prevents the data source from getting deallocated while the
    // frame is not attached?
}

void WebDocumentLoaderMac::increaseLoadCount(WebCore::ResourceLoaderIdentifier identifier)
{
    ASSERT(m_dataSource);

    if (m_loadingResources.contains(identifier))
        return;
    m_loadingResources.add(identifier);

    retainDataSource();
}

void WebDocumentLoaderMac::decreaseLoadCount(WebCore::ResourceLoaderIdentifier identifier)
{
    auto it = m_loadingResources.find(identifier);
    
    // It is valid for a load to be cancelled before it's started.
    if (it == m_loadingResources.end())
        return;
    
    m_loadingResources.remove(it);
    
    if (m_loadingResources.isEmpty()) {
        m_resourceLoadDelegate = 0;
        m_downloadDelegate = 0;
        if (!frame())
            releaseDataSource();
    }
}

void WebDocumentLoaderMac::retainDataSource()
{
    if (m_isDataSourceRetained || !m_dataSource)
        return;
    m_isDataSourceRetained = true;
    CFRetain(m_dataSource);
}

void WebDocumentLoaderMac::releaseDataSource()
{
    if (!m_isDataSourceRetained)
        return;
    ASSERT(m_dataSource);
    m_isDataSourceRetained = false;
    CFRelease(m_dataSource);
}

void WebDocumentLoaderMac::detachDataSource()
{
    ASSERT(!m_isDataSourceRetained);
    m_dataSource = nil;
}
