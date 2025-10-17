/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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
#import "ServicesController.h"

#if ENABLE(SERVICE_CONTROLS)

#import "WebProcessMessages.h"
#import "WebProcessPool.h"
#import <pal/spi/cocoa/NSExtensionSPI.h>
#import <pal/spi/mac/NSSharingServicePickerSPI.h>
#import <pal/spi/mac/NSSharingServiceSPI.h>
#import <wtf/BlockPtr.h>
#import <wtf/NeverDestroyed.h>

namespace WebKit {

ServicesController& ServicesController::singleton()
{
    static NeverDestroyed<ServicesController> sharedController;
    return sharedController;
}

ServicesController::ServicesController()
    : m_refreshQueue(dispatch_queue_create("com.apple.WebKit.ServicesController", DISPATCH_QUEUE_SERIAL))
    , m_hasPendingRefresh(false)
    , m_hasImageServices(false)
    , m_hasSelectionServices(false)
    , m_hasRichContentServices(false)
{
    refreshExistingServices();

    auto refreshCallback = [this](NSArray *, NSError *) {
        // We coalese refreshes from the notification callbacks because they can come in small batches.
        refreshExistingServices(false);
    };

    auto extensionAttributes = @{ @"NSExtensionPointName" : @"com.apple.services" };
    m_extensionWatcher = [NSExtension beginMatchingExtensionsWithAttributes:extensionAttributes completion:refreshCallback];
    auto uiExtensionAttributes = @{ @"NSExtensionPointName" : @"com.apple.ui-services" };
    m_uiExtensionWatcher = [NSExtension beginMatchingExtensionsWithAttributes:uiExtensionAttributes completion:refreshCallback];
}

static void hasCompatibleServicesForItems(dispatch_group_t group, NSArray *items, WTF::Function<void(bool)>&& completionHandler)
{
    NSSharingServiceMask servicesMask = NSSharingServiceMaskViewer | NSSharingServiceMaskEditor;

    dispatch_group_enter(group);
    [NSSharingService getSharingServicesForItems:items mask:servicesMask completion:makeBlockPtr([completionHandler = WTFMove(completionHandler), group](NSArray *services) {
        completionHandler(services.count);
        dispatch_group_leave(group);
    }).get()];
}

void ServicesController::refreshExistingServices(bool refreshImmediately)
{
    if (m_hasPendingRefresh)
        return;

    m_hasPendingRefresh = true;

    auto refreshTime = dispatch_time(DISPATCH_TIME_NOW, refreshImmediately ? 0 : (int64_t)(1 * NSEC_PER_SEC));
    dispatch_after(refreshTime, m_refreshQueue, ^{
        auto serviceLookupGroup = adoptOSObject(dispatch_group_create());

        static NSImage *image { [[NSImage alloc] init] };
        hasCompatibleServicesForItems(serviceLookupGroup.get(), @[ image ], [this] (bool hasServices) {
            m_hasImageServices = hasServices;
        });

        static NSAttributedString *attributedString { [[NSAttributedString alloc] initWithString:@"a"] };
        hasCompatibleServicesForItems(serviceLookupGroup.get(), @[ attributedString ], [this] (bool hasServices) {
            m_hasSelectionServices = hasServices;
        });

        static NeverDestroyed<RetainPtr<NSAttributedString>> attributedStringWithRichContent;
        static std::once_flag attributedStringWithRichContentOnceFlag;
        std::call_once(attributedStringWithRichContentOnceFlag, [&] {
            WorkQueue::main().dispatchSync([&] {
                auto attachment = adoptNS([[NSTextAttachment alloc] init]);
                auto cell = adoptNS([[NSTextAttachmentCell alloc] initImageCell:image]);
                [attachment setAttachmentCell:cell.get()];
                auto richString = adoptNS([[NSAttributedString attributedStringWithAttachment:attachment.get()] mutableCopy]);
                [richString appendAttributedString:attributedString];
                attributedStringWithRichContent.get() = WTFMove(richString);
            });
        });

        hasCompatibleServicesForItems(serviceLookupGroup.get(), @[ attributedStringWithRichContent.get().get() ], [this] (bool hasServices) {
            m_hasRichContentServices = hasServices;
        });

        dispatch_group_notify(serviceLookupGroup.get(), dispatch_get_main_queue(), makeBlockPtr([this] {
            bool availableServicesChanged = (m_lastSentHasImageServices != m_hasImageServices) || (m_lastSentHasSelectionServices != m_hasSelectionServices) || (m_lastSentHasRichContentServices != m_hasRichContentServices);

            m_lastSentHasSelectionServices = m_hasSelectionServices;
            m_lastSentHasImageServices = m_hasImageServices;
            m_lastSentHasRichContentServices = m_hasRichContentServices;

            if (availableServicesChanged) {
                for (auto& processPool : WebProcessPool::allProcessPools())
                    processPool->sendToAllProcesses(Messages::WebProcess::SetEnabledServices(m_hasImageServices, m_hasSelectionServices, m_hasRichContentServices));
            }

            m_hasPendingRefresh = false;
        }).get());
    });
}

} // namespace WebKit

#endif // ENABLE(SERVICE_CONTROLS)
