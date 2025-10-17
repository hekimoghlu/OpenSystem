/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
#import <WebKitLegacy/WebBasePluginPackage.h>
#import <WebKitLegacy/WebPluginContainerCheck.h>

@class WebFrame;
@class WebHTMLView;
@class WebPluginPackage;
@class WebView;
@class WebDataSource;
#if PLATFORM(IOS_FAMILY)
@class CALayer;
#endif

@interface WebPluginController : NSObject <WebPluginManualLoader, WebPluginContainerCheckController>
{
    NSView *_documentView;
    WebDataSource *_dataSource;
    NSMutableArray *_views;
    BOOL _started;
    NSMutableSet *_checksInProgress;
}

- (NSView *)plugInViewWithArguments:(NSDictionary *)arguments fromPluginPackage:(WebPluginPackage *)plugin;
+ (BOOL)isPlugInView:(NSView *)view;

- (id)initWithDocumentView:(NSView *)view;

- (void)setDataSource:(WebDataSource *)dataSource;

- (void)addPlugin:(NSView *)view;
- (void)destroyPlugin:(NSView *)view;

#if PLATFORM(IOS_FAMILY)
+ (void)addPlugInView:(NSView *)view;
- (BOOL)plugInsAreRunning;
- (CALayer *)superlayerForPluginView:(NSView *)view;
#endif
- (void)startAllPlugins;
- (void)stopAllPlugins;
- (void)destroyAllPlugins;
#if PLATFORM(IOS_FAMILY)
- (void)stopPluginsForPageCache;
- (void)restorePluginsFromCache;
#endif

- (WebFrame *)webFrame;
- (WebView *)webView;

- (NSString *)URLPolicyCheckReferrer;

@end
