/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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
#import "AXObjectCache.h"

#if PLATFORM(IOS_FAMILY)

#import "AccessibilityObject.h"
#import "Chrome.h"
#import "RenderObject.h"
#import "WebAccessibilityObjectWrapperIOS.h"
#import <wtf/RetainPtr.h>

namespace WebCore {

void AXObjectCache::attachWrapper(AccessibilityObject& object)
{
    RetainPtr<AccessibilityObjectWrapper> wrapper = adoptNS([[WebAccessibilityObjectWrapper alloc] initWithAccessibilityObject:object]);
    object.setWrapper(wrapper.get());
}

ASCIILiteral AXObjectCache::notificationPlatformName(AXNotification notification)
{
    ASCIILiteral name;

    switch (notification) {
    case AXNotification::ActiveDescendantChanged:
    case AXNotification::FocusedUIElementChanged:
        name = "AXFocusChanged"_s;
        break;
    case AXNotification::ImageOverlayChanged:
        name = "AXImageOverlayChanged"_s;
        break;
    case AXNotification::PageScrolled:
        name = "AXPageScrolled"_s;
        break;
    case AXNotification::SelectedCellsChanged:
        name = "AXSelectedCellsChanged"_s;
        break;
    case AXNotification::SelectedTextChanged:
        name = "AXSelectedTextChanged"_s;
        break;
    case AXNotification::LiveRegionChanged:
    case AXNotification::LiveRegionCreated:
        name = "AXLiveRegionChanged"_s;
        break;
    case AXNotification::InvalidStatusChanged:
        name = "AXInvalidStatusChanged"_s;
        break;
    case AXNotification::CheckedStateChanged:
    case AXNotification::ValueChanged:
        name = "AXValueChanged"_s;
        break;
    case AXNotification::ExpandedChanged:
        name = "AXExpandedChanged"_s;
        break;
    case AXNotification::CurrentStateChanged:
        name = "AXCurrentStateChanged"_s;
        break;
    case AXNotification::SortDirectionChanged:
        name = "AXSortDirectionChanged"_s;
        break;
    case AXNotification::AnnouncementRequested:
        name = "AXAnnouncementRequested"_s;
        break;
    default:
        break;
    }

    return name;
}

void AXObjectCache::relayNotification(const String& notificationName, RetainPtr<NSData> notificationData)
{
    if (RefPtr page = document() ? document()->page() : nullptr)
        page->chrome().relayAccessibilityNotification(notificationName, notificationData);
}

void AXObjectCache::postPlatformNotification(AccessibilityObject& object, AXNotification notification)
{
    auto stringNotification = notificationPlatformName(notification);
    if (stringNotification.isEmpty())
        return;

    auto notificationName = stringNotification.createNSString();
    [object.wrapper() accessibilityOverrideProcessNotification:notificationName.get() notificationData:nil];

    // To simulate AX notifications for LayoutTests on the simulator, call
    // the wrapper's accessibilityPostedNotification.
    [object.wrapper() accessibilityPostedNotification:notificationName.get()];
}

void AXObjectCache::postPlatformAnnouncementNotification(const String& message)
{
    auto notificationName = notificationPlatformName(AXNotification::AnnouncementRequested).createNSString();
    NSString *nsMessage = static_cast<NSString *>(message);
    if (RefPtr root = getOrCreate(m_document->view())) {
        [root->wrapper() accessibilityOverrideProcessNotification:notificationName.get() notificationData:[nsMessage dataUsingEncoding:NSUTF8StringEncoding]];

        // To simulate AX notifications for LayoutTests on the simulator, call
        // the wrapper's accessibilityPostedNotification.
        [root->wrapper() accessibilityPostedNotification:notificationName.get() userInfo:@{ notificationName.get() : nsMessage }];
    }
}

void AXObjectCache::postTextStateChangePlatformNotification(AccessibilityObject* object, const AXTextStateChangeIntent&, const VisibleSelection&)
{
    if (object)
        postPlatformNotification(*object, AXNotification::SelectedTextChanged);
}

void AXObjectCache::postTextStateChangePlatformNotification(AccessibilityObject* object, AXTextEditType, const String&, const VisiblePosition&)
{
    if (object)
        postPlatformNotification(*object, AXNotification::ValueChanged);
}

void AXObjectCache::postTextReplacementPlatformNotification(AccessibilityObject* object, AXTextEditType, const String&, AXTextEditType, const String&, const VisiblePosition&)
{
    if (object)
        postPlatformNotification(*object, AXNotification::ValueChanged);
}

void AXObjectCache::postTextReplacementPlatformNotificationForTextControl(AccessibilityObject* object, const String&, const String&)
{
    if (object)
        postPlatformNotification(*object, AXNotification::ValueChanged);
}

void AXObjectCache::frameLoadingEventPlatformNotification(AccessibilityObject* axFrameObject, AXLoadingEvent loadingEvent)
{
    if (!axFrameObject)
        return;

    if (loadingEvent == AXLoadingEvent::Finished && axFrameObject->document() == axFrameObject->topDocument())
        postPlatformNotification(*axFrameObject, AXNotification::LoadComplete);
}

void AXObjectCache::platformHandleFocusedUIElementChanged(Element*, Element* newElement)
{
    postNotification(newElement, AXNotification::FocusedUIElementChanged);
}

void AXObjectCache::handleScrolledToAnchor(const Node&)
{
}

void AXObjectCache::platformPerformDeferredCacheUpdate()
{
}

}

#endif // PLATFORM(IOS_FAMILY)
