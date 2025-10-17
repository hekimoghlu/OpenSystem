/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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

#include <WebCore/LoaderStrategy.h>
#include <WebCore/PasteboardStrategy.h>
#include <WebCore/PlatformStrategies.h>
#include <WebCore/PushStrategy.h>

namespace WebKit {

class WebPlatformStrategies :
    public WebCore::PlatformStrategies,
    private WebCore::PasteboardStrategy
#if ENABLE(DECLARATIVE_WEB_PUSH)
    , public WebCore::PushStrategy
#endif
{
    friend NeverDestroyed<WebPlatformStrategies>;
public:
    static void initialize();
    
private:
    WebPlatformStrategies();
    
    // WebCore::PlatformStrategies
    WebCore::LoaderStrategy* createLoaderStrategy() override;
    WebCore::PasteboardStrategy* createPasteboardStrategy() override;
    WebCore::MediaStrategy* createMediaStrategy() override;
    WebCore::BlobRegistry* createBlobRegistry() override;
#if ENABLE(DECLARATIVE_WEB_PUSH)
    WebCore::PushStrategy* createPushStrategy() override;
#endif

    // WebCore::PasteboardStrategy
#if PLATFORM(IOS_FAMILY)
    void writeToPasteboard(const WebCore::PasteboardWebContent&, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    void writeToPasteboard(const WebCore:: PasteboardURL&, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    void writeToPasteboard(const WebCore::PasteboardImage&, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    void writeToPasteboard(const String& pasteboardType, const String&, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    void updateSupportedTypeIdentifiers(const Vector<String>& identifiers, const String& pasteboardName, const WebCore::PasteboardContext*) override;
#endif
#if PLATFORM(COCOA)
    int getNumberOfFiles(const String& pasteboardName, const WebCore::PasteboardContext*) override;
    void getTypes(Vector<String>& types, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    RefPtr<WebCore::SharedBuffer> bufferForType(const String& pasteboardType, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    void getPathnamesForType(Vector<String>& pathnames, const String& pasteboardType, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    String stringForType(const String& pasteboardType, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    Vector<String> allStringsForType(const String& pasteboardType, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    int64_t changeCount(const String& pasteboardName, const WebCore::PasteboardContext*) override;
    WebCore::Color color(const String& pasteboardName, const WebCore::PasteboardContext*) override;
    URL url(const String& pasteboardName, const WebCore::PasteboardContext*) override;

    int64_t addTypes(const Vector<String>& pasteboardTypes, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    int64_t setTypes(const Vector<String>& pasteboardTypes, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    int64_t setBufferForType(WebCore::SharedBuffer*, const String& pasteboardType, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    int64_t setURL(const WebCore::PasteboardURL&, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    int64_t setColor(const WebCore::Color&, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    int64_t setStringForType(const String&, const String& pasteboardType, const String& pasteboardName, const WebCore::PasteboardContext*) override;

    bool containsURLStringSuitableForLoading(const String& pasteboardName, const WebCore::PasteboardContext*) override;
    String urlStringSuitableForLoading(const String& pasteboardName, String& title, const WebCore::PasteboardContext*) override;
#endif
#if PLATFORM(GTK)
    Vector<String> types(const String& pasteboardName) override;
    String readTextFromClipboard(const String& pasteboardName) override;
    Vector<String> readFilePathsFromClipboard(const String& pasteboardName) override;
    RefPtr<WebCore::SharedBuffer> readBufferFromClipboard(const String& pasteboardName, const String& pasteboardType) override;
    void writeToClipboard(const String& pasteboardName, WebCore::SelectionData&&) override;
    void clearClipboard(const String& pasteboardName) override;
    int64_t changeCount(const String& pasteboardName) override;
#endif
#if USE(LIBWPE)
    void getTypes(Vector<String>& types) override;
    void writeToPasteboard(const WebCore::PasteboardWebContent&) override;
    void writeToPasteboard(const String& pasteboardType, const String&) override;
#endif

    String readStringFromPasteboard(size_t index, const String& pasteboardType, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    RefPtr<WebCore::SharedBuffer> readBufferFromPasteboard(std::optional<size_t> index, const String& pasteboardType, const String& pasteboardName, const WebCore::PasteboardContext*) override;
    URL readURLFromPasteboard(size_t index, const String& pasteboardName, String& title, const WebCore::PasteboardContext*) override;
    int getPasteboardItemsCount(const String& pasteboardName, const WebCore::PasteboardContext*) override;
    std::optional<WebCore::PasteboardItemInfo> informationForItemAtIndex(size_t index, const String& pasteboardName, int64_t changeCount, const WebCore::PasteboardContext*) override;
    std::optional<Vector<WebCore::PasteboardItemInfo>> allPasteboardItemInfo(const String& pasteboardName, int64_t changeCount, const WebCore::PasteboardContext*) override;
    Vector<String> typesSafeForDOMToReadAndWrite(const String& pasteboardName, const String& origin, const WebCore::PasteboardContext*) override;
    int64_t writeCustomData(const Vector<WebCore::PasteboardCustomData>&, const String&, const WebCore::PasteboardContext*) override;
    bool containsStringSafeForDOMToReadForType(const String&, const String& pasteboardName, const WebCore::PasteboardContext*) override;

    // WebCore::PushStrategy
#if ENABLE(DECLARATIVE_WEB_PUSH)
    void windowSubscribeToPushService(const URL& scope, const Vector<uint8_t>& applicationServerKey, SubscribeToPushServiceCallback&&) override;
    void windowUnsubscribeFromPushService(const URL& scope, std::optional<WebCore::PushSubscriptionIdentifier>, UnsubscribeFromPushServiceCallback&&) override;
    void windowGetPushSubscription(const URL& scope, GetPushSubscriptionCallback&&) override;
    void windowGetPushPermissionState(const URL& scope, GetPushPermissionStateCallback&&) override;
#endif
};

} // namespace WebKit
