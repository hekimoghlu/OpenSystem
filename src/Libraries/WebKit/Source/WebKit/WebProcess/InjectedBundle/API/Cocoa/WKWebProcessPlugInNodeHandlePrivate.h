/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#import <WebKit/WKWebProcessPlugInNodeHandle.h>

typedef NS_ENUM(NSInteger, _WKAutoFillButtonType) {
    _WKAutoFillButtonTypeNone,
    _WKAutoFillButtonTypeCredentials,
    _WKAutoFillButtonTypeContacts,
    _WKAutoFillButtonTypeStrongPassword,
    _WKAutoFillButtonTypeCreditCard WK_API_AVAILABLE(macos(10.14.4), ios(12.2)),
    _WKAutoFillButtonTypeLoading WK_API_AVAILABLE(macos(13.0), ios(16.0)),
} WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

@interface WKWebProcessPlugInNodeHandle (WKPrivate)

- (BOOL)isHTMLInputElementAutoFillButtonEnabled WK_API_AVAILABLE(macos(10.13.4), ios(11.3));
- (void)setHTMLInputElementAutoFillButtonEnabledWithButtonType:(_WKAutoFillButtonType)autoFillButtonType WK_API_AVAILABLE(macos(10.13.4), ios(11.3));
- (_WKAutoFillButtonType)htmlInputElementAutoFillButtonType WK_API_AVAILABLE(macos(10.13.4), ios(11.3));
- (_WKAutoFillButtonType)htmlInputElementLastAutoFillButtonType WK_API_AVAILABLE(macos(10.13.4), ios(11.3));

@end
