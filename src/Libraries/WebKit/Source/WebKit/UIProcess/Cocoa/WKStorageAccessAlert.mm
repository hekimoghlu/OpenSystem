/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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
#import "WKStorageAccessAlert.h"

#import <wtf/HashMap.h>

#if PLATFORM(IOS_FAMILY)
#import "UIKitSPI.h"
#import "UIKitUtilities.h"
#endif

#if PLATFORM(COCOA) && !PLATFORM(WATCHOS) && !PLATFORM(APPLETV)

#import "WKWebViewInternal.h"
#import <WebCore/LocalizedStrings.h>
#import <WebCore/NetworkStorageSession.h>
#import <WebCore/RegistrableDomain.h>
#import <wtf/BlockPtr.h>

#if PLATFORM(MAC)
@interface _WKSSOSiteList : NSObject<NSTableViewDataSource, NSTableViewDelegate>
- (instancetype)initWithSiteList:(NSArray<NSString *> *)siteList withAlert:(RetainPtr<NSAlert>)alert withTableView:(RetainPtr<NSTableView>)tableView;
- (IBAction)toggleTableViewContents:(NSButton *)sender;
@end

@implementation _WKSSOSiteList {
    RetainPtr<NSArray<NSString *>> _siteList;
    RetainPtr<NSAlert> _alert;
    RetainPtr<NSTableView> _tableView;
}

- (instancetype)initWithSiteList:(NSArray<NSString *> *)siteList withAlert:(RetainPtr<NSAlert>)alert withTableView:(RetainPtr<NSTableView>)tableView
{
    if (!(self = [super init]))
        return self;

    _siteList = adoptNS([[NSArray alloc] initWithArray:siteList copyItems:YES]);
    _alert = alert;
    _tableView = tableView;
    return self;
}

- (NSInteger)numberOfRowsInTableView:(NSTableView *)aTableView
{
    return _siteList.get().count;
}

- (NSView *)tableView:(NSTableView *)tableView viewForTableColumn:(NSTableColumn *)tableColumn row:(NSInteger)row
{
    NSString *viewIdenfier = [NSString stringWithFormat:@"row%ldIdentifier", (long)row];
    RetainPtr result = [tableView makeViewWithIdentifier:viewIdenfier owner:self];
    if (!result) {
        result = adoptNS([[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, 100, 300)]);
        result.get().identifier = viewIdenfier;
    }
    [result setEditable:NO];
    [result setDrawsBackground:NO];

    if ((NSUInteger)row < [_siteList count]) {
        [[result textStorage].mutableString setString:_siteList.get()[row]];
        [result textStorage].font = [NSFont systemFontOfSize:[NSFont smallSystemFontSize]];
        [result textStorage].foregroundColor = NSColor.whiteColor;
    }
    return result.autorelease();
}

- (IBAction)toggleTableViewContents:(NSButton *)sender
{
    _tableView.get().hidden = (sender.state != NSControlStateValueOn);
    NSRect frame = _alert.get().window.frame;
    if (_tableView.get().hidden)
        frame.size.height -= _tableView.get().frame.size.height;
    else
        frame.size.height += _tableView.get().frame.size.height;
    [_alert.get().window setFrame:frame display:YES];
    [_alert layout];
}
@end
#endif

namespace WebKit {

void presentStorageAccessAlert(WKWebView *webView, const WebCore::RegistrableDomain& requesting, const WebCore::RegistrableDomain& current, CompletionHandler<void(bool)>&& completionHandler)
{
    auto requestingDomain = requesting.string().createCFString();
    auto currentDomain = current.string().createCFString();

#if PLATFORM(MAC)
    NSString *alertTitle = [NSString stringWithFormat:WEB_UI_NSSTRING(@"Do you want to allow â€œ%@â€ to use cookies and website data while browsing â€œ%@â€?", @"Message for requesting cross-site cookie and website data access."), requestingDomain.get(), currentDomain.get()];
#else
    NSString *alertTitle = [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow â€œ%@â€ to use cookies and website data while browsing â€œ%@â€?", @"Message for requesting cross-site cookie and website data access."), requestingDomain.get(), currentDomain.get()];
#endif

    NSString *informativeText = [NSString stringWithFormat:WEB_UI_NSSTRING(@"This will allow â€œ%@â€ to track your activity.", @"Informative text for requesting cross-site cookie and website data access."), requestingDomain.get()];

    displayStorageAccessAlert(webView, alertTitle, informativeText, nil, nil, WTFMove(completionHandler));
}

void presentStorageAccessAlertQuirk(WKWebView *webView, const WebCore::RegistrableDomain& firstRequesting, const WebCore::RegistrableDomain& secondRequesting, const WebCore::RegistrableDomain& current, CompletionHandler<void(bool)>&& completionHandler)
{
    auto firstRequestingDomain = firstRequesting.string().createCFString();
    auto secondRequestingDomain = secondRequesting.string().createCFString();
    auto currentDomain = current.string().createCFString();

#if PLATFORM(MAC)
    NSString *alertTitle = [NSString stringWithFormat:WEB_UI_NSSTRING(@"Do you want to allow â€œ%@â€ and â€œ%@â€ to use cookies and website data while browsing â€œ%@â€?", @"Message for requesting cross-site cookie and website data access."), firstRequestingDomain.get(), secondRequestingDomain.get(), currentDomain.get()];
#else
    NSString *alertTitle = [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow â€œ%@â€ and â€œ%@â€ to use cookies and website data while browsing â€œ%@â€?", @"Message for requesting cross-site cookie and website data access."), firstRequestingDomain.get(), secondRequestingDomain.get(), currentDomain.get()];
#endif

    NSString *informativeText = [NSString stringWithFormat:WEB_UI_NSSTRING(@"This will allow â€œ%@â€ and â€œ%@â€ to track your activity.", @"Informative text for requesting cross-site cookie and website data access."), firstRequestingDomain.get(), secondRequestingDomain.get()];

    displayStorageAccessAlert(webView, alertTitle, informativeText, nil, nil, WTFMove(completionHandler));
}

void presentStorageAccessAlertSSOQuirk(WKWebView *webView, const String& organizationName, const HashMap<WebCore::RegistrableDomain, Vector<WebCore::RegistrableDomain>>& domainPairings, CompletionHandler<void(bool)>&& completionHandler)
{
    NSString *alertTitle = [NSString stringWithFormat:WEB_UI_NSSTRING(@"Allow related %@ websites to share cookies and website data?", @"Message for requesting cross-site cookie and website data access."), organizationName.createCFString().get()];

    NSString *informativeText;;
    NSString *relatedWebsitesString;
    NSMutableArray<NSString *> *accessoryTextList;

    HashSet<String> allDomains;
    for (auto&& domains : domainPairings) {
        allDomains.add(domains.key.string());
        for (auto&& subFrameDomain : domains.value)
            allDomains.add(subFrameDomain.string());
    }

    if (allDomains.size() < 2)  {
        completionHandler(true);
        return;
    }

    Vector<String> uniqueDomainList = copyToVector(allDomains);
    std::sort(uniqueDomainList.begin(), uniqueDomainList.end(), WTF::codePointCompareLessThan);

    if (uniqueDomainList.size() < 4) {
        auto lastSite = uniqueDomainList.takeLast();
        StringBuilder initialListOfSites;
        initialListOfSites.append(makeStringByJoining(uniqueDomainList.span(), ", "_s));
        if (uniqueDomainList.size() == 2)
            initialListOfSites.append(","_s);

        informativeText = [NSString stringWithFormat:WEB_UI_NSSTRING(@"Using the same cookies and website data is required for %s and %s to work correctly, but could make it easier to track your browsing across these websites.", @"Informative text for requesting cross-site cookie and website data access for two sites"), initialListOfSites.toString().utf8().data(), lastSite.utf8().data()];
        relatedWebsitesString = nil;
        accessoryTextList = nil;
    } else {
        informativeText = [NSString stringWithFormat:WEB_UI_NSSTRING(@"Using the same cookies and website data is required for %s, %s, and %lu other websites to work correctly, but could make it easier to track your browsing across these websites.", @"Informative text for requesting cross-site cookie and website data access for four or more sites."), uniqueDomainList[0].utf8().data(), uniqueDomainList[1].utf8().data(), uniqueDomainList.size() - 2];

        relatedWebsitesString = [NSString stringWithFormat:WEB_UI_NSSTRING(@"Related %@ websites", @"Label describing the list of related websites controlled by the same organization"), organizationName.createCFString().get()];
        accessoryTextList = [NSMutableArray arrayWithCapacity:uniqueDomainList.size()];
        for (const auto& domains : uniqueDomainList)
            [accessoryTextList addObject:domains];
    }

    displayStorageAccessAlert(webView, alertTitle, informativeText, relatedWebsitesString, accessoryTextList, WTFMove(completionHandler));
}

void displayStorageAccessAlert(WKWebView *webView, NSString *alertTitle, NSString *informativeText, NSString *accessoryLabel, NSArray<NSString *> *accessoryTextList, CompletionHandler<void(bool)>&& completionHandler)
{
    auto completionBlock = makeBlockPtr([completionHandler = WTFMove(completionHandler)](bool shouldAllow) mutable {
        completionHandler(shouldAllow);
    });

    NSString *allowButtonString = WEB_UI_STRING_KEY(@"Allow", "Allow (cross-site cookie and website data access)", @"Button title in Storage Access API prompt");
    NSString *doNotAllowButtonString = WEB_UI_STRING_KEY(@"Donâ€™t Allow", "Donâ€™t Allow (cross-site cookie and website data access)", @"Button title in Storage Access API prompt");

#if PLATFORM(MAC)
    auto alert = adoptNS([NSAlert new]);
    [alert setMessageText:alertTitle];
    [alert setInformativeText:informativeText];
    RetainPtr<_WKSSOSiteList> ssoSiteList;
    if (accessoryTextList.count) {
        RetainPtr disclosureButton = adoptNS([[NSButton alloc] init]);
        disclosureButton.get().title = @"";
        disclosureButton.get().bezelStyle = NSBezelStyleDisclosure;
        [disclosureButton setButtonType:NSButtonTypePushOnPushOff];

        NSTextField *relatedWebsitesLabel = [NSTextField labelWithString:accessoryLabel];
        NSStackView *disclosureStackView = [NSStackView stackViewWithViews:@[disclosureButton.get(), relatedWebsitesLabel]];

        RetainPtr siteListTableView = adoptNS([[NSTableView alloc] initWithFrame:NSMakeRect(0, 0, 1000, 1000)]);
        siteListTableView.get().allowsTypeSelect = NO;
        siteListTableView.get().enabled = NO;
        siteListTableView.get().usesSingleLineMode = YES;
        siteListTableView.get().hidden = YES;
        [siteListTableView addTableColumn:adoptNS([[NSTableColumn alloc] initWithIdentifier:@"columnIdentifier"]).get()];

        ssoSiteList = adoptNS([[_WKSSOSiteList alloc] initWithSiteList:accessoryTextList withAlert:alert withTableView:siteListTableView]);
        siteListTableView.get().dataSource = ssoSiteList.get();
        siteListTableView.get().delegate = ssoSiteList.get();

        NSStackView *accessoryStackView = [NSStackView stackViewWithViews:@[disclosureStackView, siteListTableView.get()]];
        accessoryStackView.orientation = NSUserInterfaceLayoutOrientationVertical;
        disclosureButton.get().target = ssoSiteList.get();
        disclosureButton.get().action = @selector(toggleTableViewContents:);

        NSRect frame = alert.get().window.frame;
        frame.size.width += 100.;
        [alert.get().window setFrame:frame display:YES];

        [alert setAccessoryView:accessoryStackView];
        [alert layout];
    }
    [alert addButtonWithTitle:allowButtonString];
    [alert addButtonWithTitle:doNotAllowButtonString];
    [alert beginSheetModalForWindow:webView.window completionHandler:makeBlockPtr([ssoSiteList, completionBlock](NSModalResponse returnCode) {
        auto shouldAllow = returnCode == NSAlertFirstButtonReturn;
        completionBlock(shouldAllow);
    }).get()];
#else
    auto alert = WebKit::createUIAlertController(alertTitle, informativeText);

    UIAlertAction* allowAction = [UIAlertAction actionWithTitle:allowButtonString style:UIAlertActionStyleCancel handler:[completionBlock](UIAlertAction *action) {
        completionBlock(true);
    }];

    UIAlertAction* doNotAllowAction = [UIAlertAction actionWithTitle:doNotAllowButtonString style:UIAlertActionStyleDefault handler:[completionBlock](UIAlertAction *action) {
        completionBlock(false);
    }];

    [alert addAction:doNotAllowAction];
    [alert addAction:allowAction];

    [webView._wk_viewControllerForFullScreenPresentation presentViewController:alert.get() animated:YES completion:nil];
#endif
}

}

#endif // PLATFORM(COCOA) && !PLATFORM(WATCHOS) && !PLATFORM(APPLETV)
