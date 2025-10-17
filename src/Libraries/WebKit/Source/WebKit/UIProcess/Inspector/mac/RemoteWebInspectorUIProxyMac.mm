/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#import "RemoteWebInspectorUIProxy.h"

#if PLATFORM(MAC) && ENABLE(REMOTE_INSPECTOR)

#import "GlobalFindInPageState.h"
#import "MessageSenderInlines.h"
#import "RemoteWebInspectorUIMessages.h"
#import "RemoteWebInspectorUIProxyMessages.h"
#import "WKInspectorViewController.h"
#import "WKWebViewInternal.h"
#import "WebInspectorUIProxy.h"
#import "WebPageGroup.h"
#import "WebPageProxy.h"
#import "_WKInspectorConfigurationInternal.h"
#import <SecurityInterface/SFCertificatePanel.h>
#import <SecurityInterface/SFCertificateView.h>
#import <WebCore/CertificateInfo.h>
#import <WebCore/Color.h>
#import <WebKit/WKFrameInfo.h>
#import <WebKit/WKNavigationAction.h>
#import <WebKit/WKNavigationDelegate.h>
#import <wtf/BlockPtr.h>
#import <wtf/text/Base64.h>

@interface WKRemoteWebInspectorUIProxyObjCAdapter : NSObject <NSWindowDelegate, WKInspectorViewControllerDelegate> {
    WeakPtr<WebKit::RemoteWebInspectorUIProxy> _inspectorProxy;
}
- (instancetype)initWithRemoteWebInspectorUIProxy:(WebKit::RemoteWebInspectorUIProxy*)inspectorProxy;
@end

@implementation WKRemoteWebInspectorUIProxyObjCAdapter

- (RefPtr<WebKit::RemoteWebInspectorUIProxy>)protectedInspectorProxy
{
    return _inspectorProxy.get();
}

- (NSRect)window:(NSWindow *)window willPositionSheet:(NSWindow *)sheet usingRect:(NSRect)rect
{
    if (_inspectorProxy)
        return NSMakeRect(0, _inspectorProxy->sheetRect().height(), _inspectorProxy->sheetRect().width(), 0);
    return rect;
}

- (instancetype)initWithRemoteWebInspectorUIProxy:(WebKit::RemoteWebInspectorUIProxy*)inspectorProxy
{
    if (!(self = [super init]))
        return nil;

    _inspectorProxy = inspectorProxy;

    return self;
}

- (void)inspectorWKWebViewDidBecomeActive:(WKInspectorViewController *)inspectorViewController
{
    self.protectedInspectorProxy->didBecomeActive();
}

- (void)inspectorViewControllerInspectorDidCrash:(WKInspectorViewController *)inspectorViewController
{
    self.protectedInspectorProxy->closeFromCrash();
}

- (BOOL)inspectorViewControllerInspectorIsUnderTest:(WKInspectorViewController *)inspectorViewController
{
    return self.protectedInspectorProxy->isUnderTest();
}

@end

namespace WebKit {
using namespace WebCore;

WKWebView *RemoteWebInspectorUIProxy::webView() const
{
    return m_inspectorView.get().webView;
}

void RemoteWebInspectorUIProxy::didBecomeActive()
{
    protectedInspectorPage()->protectedLegacyMainFrameProcess()->send(Messages::RemoteWebInspectorUI::UpdateFindString(WebKit::stringForFind()), m_inspectorPage->webPageIDInMainFrameProcess());
}

WebPageProxy* RemoteWebInspectorUIProxy::platformCreateFrontendPageAndWindow()
{
    m_objCAdapter = adoptNS([[WKRemoteWebInspectorUIProxyObjCAdapter alloc] initWithRemoteWebInspectorUIProxy:this]);

    Ref<API::InspectorConfiguration> configuration = m_client->configurationForRemoteInspector(*this);
    m_inspectorView = adoptNS([[WKInspectorViewController alloc] initWithConfiguration: WebKit::wrapper(configuration) inspectedPage:nullptr]);
    [m_inspectorView.get() setDelegate:m_objCAdapter.get()];

    m_window = WebInspectorUIProxy::createFrontendWindow(NSZeroRect, WebInspectorUIProxy::InspectionTargetType::Remote);
    [m_window setDelegate:m_objCAdapter.get()];
    [m_window setFrameAutosaveName:@"WKRemoteWebInspectorWindowFrame"];

    NSView *contentView = m_window.get().contentView;
    [webView() setFrame:contentView.bounds];
    [contentView addSubview:webView()];

    return webView()->_page.get();
}

void RemoteWebInspectorUIProxy::platformCloseFrontendPageAndWindow()
{
    if (m_window) {
        [m_window setDelegate:nil];
        [m_window close];
        m_window = nil;
    }

    if (m_inspectorView) {
        [m_inspectorView.get() setDelegate:nil];
        m_inspectorView = nil;
    }

    if (m_objCAdapter)
        m_objCAdapter = nil;
}

void RemoteWebInspectorUIProxy::platformResetState()
{
    [NSWindow removeFrameUsingName:[m_window frameAutosaveName]];
}

void RemoteWebInspectorUIProxy::platformBringToFront()
{
    [m_window makeKeyAndOrderFront:nil];
    [m_window makeFirstResponder:webView()];
}

void RemoteWebInspectorUIProxy::platformSave(Vector<InspectorFrontendClient::SaveData>&& saveDatas, bool forceSaveAs)
{
    RetainPtr<NSString> urlCommonPrefix;
    for (auto& item : saveDatas) {
        if (!urlCommonPrefix)
            urlCommonPrefix = item.url;
        else
            urlCommonPrefix = [urlCommonPrefix commonPrefixWithString:item.url options:0];
    }
    if ([urlCommonPrefix hasSuffix:@"."])
        urlCommonPrefix = [urlCommonPrefix substringToIndex:[urlCommonPrefix length] - 1];

    RetainPtr platformURL = m_suggestedToActualURLMap.get(urlCommonPrefix.get());
    if (!platformURL) {
        platformURL = [NSURL URLWithString:urlCommonPrefix.get()];
        // The user must confirm new filenames before we can save to them.
        forceSaveAs = true;
    }

    WebInspectorUIProxy::showSavePanel(m_window.get(), platformURL.get(), WTFMove(saveDatas), forceSaveAs, [urlCommonPrefix, protectedThis = Ref { *this }] (NSURL *actualURL) {
        protectedThis->m_suggestedToActualURLMap.set(urlCommonPrefix.get(), actualURL);
    });
}

void RemoteWebInspectorUIProxy::platformLoad(const String& path, CompletionHandler<void(const String&)>&& completionHandler)
{
    if (auto contents = FileSystem::readEntireFile(path))
        completionHandler(String::adopt(WTFMove(*contents)));
    else
        completionHandler(nullString());
}

void RemoteWebInspectorUIProxy::platformPickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&& completionHandler)
{
    auto sampler = adoptNS([[NSColorSampler alloc] init]);
    [sampler.get() showSamplerWithSelectionHandler:makeBlockPtr([completionHandler = WTFMove(completionHandler)](NSColor *selectedColor) mutable {
        if (!selectedColor) {
            completionHandler(std::nullopt);
            return;
        }

        completionHandler(Color::createAndPreserveColorSpace(selectedColor.CGColor));
    }).get()];
}

void RemoteWebInspectorUIProxy::platformSetSheetRect(const FloatRect& rect)
{
    m_sheetRect = rect;
}

void RemoteWebInspectorUIProxy::platformSetForcedAppearance(InspectorFrontendClient::Appearance appearance)
{
    NSAppearance *platformAppearance;
    switch (appearance) {
    case InspectorFrontendClient::Appearance::System:
        platformAppearance = nil;
        break;

    case InspectorFrontendClient::Appearance::Light:
        platformAppearance = [NSAppearance appearanceNamed:NSAppearanceNameAqua];
        break;

    case InspectorFrontendClient::Appearance::Dark:
        platformAppearance = [NSAppearance appearanceNamed:NSAppearanceNameDarkAqua];
        break;
    }

    webView().appearance = platformAppearance;

    NSWindow *window = m_window.get();
    ASSERT(window);
    window.appearance = platformAppearance;
}

void RemoteWebInspectorUIProxy::platformStartWindowDrag()
{
    webView()._protectedPage->startWindowDrag();
}

void RemoteWebInspectorUIProxy::platformOpenURLExternally(const String& url)
{
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:url]];
}

void RemoteWebInspectorUIProxy::platformRevealFileExternally(const String& path)
{
    [[NSWorkspace sharedWorkspace] activateFileViewerSelectingURLs:@[ [NSURL URLWithString:path] ]];
}

void RemoteWebInspectorUIProxy::platformShowCertificate(const CertificateInfo& certificateInfo)
{
    ASSERT(!certificateInfo.isEmpty());

    RetainPtr<SFCertificatePanel> certificatePanel = adoptNS([[SFCertificatePanel alloc] init]);

    ASSERT(m_window);
    [certificatePanel beginSheetForWindow:m_window.get() modalDelegate:nil didEndSelector:NULL contextInfo:nullptr trust:certificateInfo.trust().get() showGroup:YES];

    // This must be called after the trust panel has been displayed, because the certificateView doesn't exist beforehand.
    SFCertificateView *certificateView = [certificatePanel certificateView];
    [certificateView setDisplayTrust:YES];
    [certificateView setEditableTrust:NO];
    [certificateView setDisplayDetails:YES];
    [certificateView setDetailsDisclosed:YES];
}

} // namespace WebKit

#endif
