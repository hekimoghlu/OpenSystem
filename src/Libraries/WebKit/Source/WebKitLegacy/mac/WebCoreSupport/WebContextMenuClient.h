/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#if ENABLE(CONTEXT_MENUS)

#import "WebSharingServicePickerController.h"
#import <WebCore/ContextMenuClient.h>
#import <WebCore/IntRect.h>
#import <wtf/TZoneMalloc.h>

@class WebSharingServicePickerController;
@class WebView;

namespace WebCore {
class Node;
}

class WebContextMenuClient : public WebCore::ContextMenuClient
#if ENABLE(SERVICE_CONTROLS)
    , public WebSharingServicePickerClient
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(WebContextMenuClient);
public:
    WebContextMenuClient(WebView *webView);
    virtual ~WebContextMenuClient();

    void downloadURL(const URL&) override;
    void searchWithGoogle(const WebCore::LocalFrame*) override;
    void lookUpInDictionary(WebCore::LocalFrame*) override;
    bool isSpeaking() const override;
    void speak(const WTF::String&) override;
    void stopSpeaking() override;
    void showContextMenu() override;

#if ENABLE(IMAGE_ANALYSIS)
    bool supportsLookUpInImages() final { return false; }
#endif

#if ENABLE(IMAGE_ANALYSIS_ENHANCEMENTS)
    bool supportsCopySubject() final { return false; }
#endif

#if ENABLE(SERVICE_CONTROLS)
    // WebSharingServicePickerClient
    void sharingServicePickerWillBeDestroyed(WebSharingServicePickerController &) override;
    WebCore::FloatRect screenRectForCurrentSharingServicePickerItem(WebSharingServicePickerController &) override;
    RetainPtr<NSImage> imageForCurrentSharingServicePickerItem(WebSharingServicePickerController &) override;
#endif

#if HAVE(TRANSLATION_UI_SERVICES)
    void handleTranslation(const WebCore::TranslationContextMenuInfo&) final;
#endif

private:
    NSMenu *contextMenuForEvent(NSEvent *, NSView *, bool& isServicesMenu);

    bool clientFloatRectForNode(WebCore::Node&, WebCore::FloatRect&) const;

#if ENABLE(SERVICE_CONTROLS)
    RetainPtr<WebSharingServicePickerController> m_sharingServicePickerController;
#else
    WebView* m_webView;
#endif
};

#endif // ENABLE(CONTEXT_MENUS)

