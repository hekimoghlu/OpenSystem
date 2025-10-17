/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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
#import <WebKitLegacy/WebFrameView.h>

@class WebDynamicScrollBarsView;
@class WebView;

@interface WebFrameView (WebInternal)

- (WebView *)_webView;
- (void)_setDocumentView:(NSView <WebDocumentView> *)view;
- (NSView <WebDocumentView> *)_makeDocumentViewForDataSource:(WebDataSource *)dataSource;
- (void)_setWebFrame:(WebFrame *)webFrame;
- (float)_verticalPageScrollDistance;
+ (NSMutableDictionary *)_viewTypesAllowImageTypeOmission:(BOOL)allowImageTypeOmission;
- (Class)_viewClassForMIMEType:(NSString *)MIMEType;
+ (Class)_viewClassForMIMEType:(NSString *)MIMEType allowingPlugins:(BOOL)allowPlugins;
+ (BOOL)_canShowMIMETypeAsHTML:(NSString *)MIMEType;
- (WebDynamicScrollBarsView *)_scrollView;
- (void)_install;
- (void)_frameSizeChanged;

@end
