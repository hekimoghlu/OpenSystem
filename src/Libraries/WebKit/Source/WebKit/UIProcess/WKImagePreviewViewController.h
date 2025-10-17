/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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

#import <UIKit/UIKit.h>

#import <wtf/RetainPtr.h>

@class _WKActivatedElementInfo;
@class _WKElementAction;

@interface WKImagePreviewViewController : UIViewController {
@private
    RetainPtr<NSArray> _imageActions;
    RetainPtr<_WKActivatedElementInfo> _activatedElementInfo;
}

- (id)initWithCGImage:(RetainPtr<CGImageRef>)image defaultActions:(RetainPtr<NSArray>)actions elementInfo:(RetainPtr<_WKActivatedElementInfo>)elementInfo;
@end

#endif // PLATFORM(IOS_FAMILY)
