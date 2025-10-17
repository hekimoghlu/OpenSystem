/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#import <WebKit/WKFoundation.h>

#if !TARGET_OS_IPHONE

#import <Cocoa/Cocoa.h>
#import <WebKit/WKDeclarationSpecifiers.h>

#if !defined(WK_UNUSED_INSTANCE_VARIABLE)
#define WK_UNUSED_INSTANCE_VARIABLE
#endif

@class WKBrowsingContextController;
@class WKProcessGroup;
@class WKViewData;

// FIXME: Remove this file once rdar://105559864 and rdar://105560420 and rdar://105560497 are complete.
// Note: rdar://105011969 tracks additional use of this file, but those are unused test utilities that are never built.
// Also, this stub needs to be kept as long as Ventura-based macOS versions are used for performance testing.

WK_CLASS_DEPRECATED_WITH_REPLACEMENT("WKWebView", macos(10.10, 10.14.4), ios(8.0, 12.2))
@interface WKView : NSView <NSTextInputClient> {
@private
    WKViewData *_data;
    WK_UNUSED_INSTANCE_VARIABLE unsigned _unused;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
@property(readonly) WKBrowsingContextController *browsingContextController;
#pragma clang diagnostic pop

@property BOOL drawsBackground;
@property BOOL drawsTransparentBackground;

@end

#endif
