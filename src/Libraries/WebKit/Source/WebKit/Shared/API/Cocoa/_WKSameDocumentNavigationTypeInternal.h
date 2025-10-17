/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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
#import "SameDocumentNavigationType.h"
#import <WebKit/_WKSameDocumentNavigationType.h>

namespace WebKit {

static _WKSameDocumentNavigationType toWKSameDocumentNavigationType(SameDocumentNavigationType navigationType)
{
    switch (navigationType) {
    case SameDocumentNavigationType::AnchorNavigation:
        return _WKSameDocumentNavigationTypeAnchorNavigation;
    case SameDocumentNavigationType::SessionStatePush:
        return _WKSameDocumentNavigationTypeSessionStatePush;
    case SameDocumentNavigationType::SessionStateReplace:
        return _WKSameDocumentNavigationTypeSessionStateReplace;
    case SameDocumentNavigationType::SessionStatePop:
        return _WKSameDocumentNavigationTypeSessionStatePop;
    }

    ASSERT_NOT_REACHED();
    return _WKSameDocumentNavigationTypeAnchorNavigation;
}

}
