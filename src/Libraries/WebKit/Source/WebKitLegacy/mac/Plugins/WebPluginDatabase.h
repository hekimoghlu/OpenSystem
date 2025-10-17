/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
#import <Foundation/Foundation.h>

#if TARGET_OS_IPHONE
#import <WebKitLegacy/WAKAppKitStubs.h>
#endif

@class WebBasePluginPackage;
@class WebFrame;

@interface WebPluginDatabase : NSObject
{
@private
    NSMutableDictionary *plugins;
    NSMutableSet *registeredMIMETypes;
    NSArray *plugInPaths;
    
    // Set of views with plugins attached
    NSMutableSet *pluginInstanceViews;
}

+ (WebPluginDatabase *)sharedDatabase;
+ (WebPluginDatabase *)sharedDatabaseIfExists;
+ (void)closeSharedDatabase; // avoids creating the database just to close it

// Plug-ins are returned in this order: New plug-in (WBPL), Mach-O Netscape
- (WebBasePluginPackage *)pluginForMIMEType:(NSString *)mimeType;
- (WebBasePluginPackage *)pluginForExtension:(NSString *)extension;

- (BOOL)isMIMETypeRegistered:(NSString *)MIMEType;

- (NSArray *)plugins;

- (void)refresh;

- (void)setPlugInPaths:(NSArray *)newPaths;

- (void)close;

#if TARGET_OS_IPHONE
- (void)addPluginInstanceView:(WAKView *)view;
- (void)removePluginInstanceView:(WAKView *)view;
#else
- (void)addPluginInstanceView:(NSView *)view;
- (void)removePluginInstanceView:(NSView *)view;
#endif
- (void)removePluginInstanceViewsFor:(WebFrame *)webFrame;
- (void)destroyAllPluginInstanceViews;
@end

@interface NSObject (WebPlugInDatabase)

+ (void)setAdditionalWebPlugInPaths:(NSArray *)path;

@end
