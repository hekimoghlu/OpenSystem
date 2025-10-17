/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "WKBaseScrollView.h"
#import <wtf/OptionSet.h>

OBJC_CLASS UIScrollView;

namespace WebCore {
class FloatRect;
class IntPoint;

enum class EventListenerRegionType : uint8_t;
enum class TouchAction : uint8_t;
}

namespace WebKit {
class RemoteLayerTreeHost;
class WebPageProxy;
}

@protocol WKNativelyInteractible <NSObject>
@end

@protocol WKContentControlled <NSObject>
@end

@interface WKCompositingView : UIView <WKContentControlled>
@end

@interface WKTransformView : WKCompositingView
@end

@interface WKBackdropView : WKCompositingView
@end

@interface WKShapeView : WKCompositingView
@end

#if HAVE(CORE_MATERIAL)
@interface WKMaterialView : WKCompositingView
@end
#endif

@interface WKUIRemoteView : _UIRemoteView <WKContentControlled>
@end

@interface WKChildScrollView : WKBaseScrollView <WKContentControlled>
@end

#if USE(APPLE_INTERNAL_SDK)
#import <WebKitAdditions/WKSeparatedModelView.h>
#endif

namespace WebKit {

OptionSet<WebCore::TouchAction> touchActionsForPoint(UIView *rootView, const WebCore::IntPoint&);
UIScrollView *findActingScrollParent(UIScrollView *, const RemoteLayerTreeHost&);

OptionSet<WebCore::EventListenerRegionType> eventListenerTypesAtPoint(UIView *rootView, const WebCore::IntPoint&);

#if ENABLE(EDITABLE_REGION)
bool mayContainEditableElementsInRect(UIView *rootView, const WebCore::FloatRect&);
#endif

}

#endif // PLATFORM(IOS_FAMILY)
