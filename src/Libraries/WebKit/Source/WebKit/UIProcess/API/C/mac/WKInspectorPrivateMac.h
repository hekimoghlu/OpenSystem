/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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
#import <TargetConditionals.h>

#if !TARGET_OS_IPHONE

#import <AppKit/NSWindow.h>
#import <WebKit/WKDeclarationSpecifiers.h>
#import <WebKit/WKInspector.h>

@class _WKInspector;

#ifdef __cplusplus
extern "C" {
#endif

const NSInteger WKInspectorViewTag = 1000;

// This class is the Web Inspector window delegate. It can be used to add interface
// actions that need to work when the Web Inspector window is key.
WK_EXPORT @interface WKWebInspectorUIProxyObjCAdapter : NSObject <NSWindowDelegate>

@property (readonly) WKInspectorRef inspectorRef;
@property (readonly) _WKInspector *inspector;

@end

#ifdef __cplusplus
}
#endif

#endif
