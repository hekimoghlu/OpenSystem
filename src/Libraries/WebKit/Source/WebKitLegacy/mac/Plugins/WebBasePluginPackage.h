/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
#import <WebCore/PluginData.h>
#import <wtf/RetainPtr.h>

typedef void (*BP_CreatePluginMIMETypesPreferencesFuncPtr)(void);

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WAKAppKitStubs.h>
#endif

@class WebPluginDatabase;

@protocol WebPluginManualLoader
- (void)pluginView:(NSView *)pluginView receivedResponse:(NSURLResponse *)response;
- (void)pluginView:(NSView *)pluginView receivedData:(NSData *)data;
- (void)pluginView:(NSView *)pluginView receivedError:(NSError *)error;
- (void)pluginViewFinishedLoading:(NSView *)pluginView;
@end

#define WebPluginExtensionsKey          @"WebPluginExtensions"
#define WebPluginDescriptionKey         @"WebPluginDescription"
#define WebPluginLocalizationNameKey    @"WebPluginLocalizationName"
#define WebPluginMIMETypesFilenameKey   @"WebPluginMIMETypesFilename"
#define WebPluginMIMETypesKey           @"WebPluginMIMETypes"
#define WebPluginNameKey                @"WebPluginName"
#define WebPluginTypeDescriptionKey     @"WebPluginTypeDescription"
#define WebPluginTypeEnabledKey         @"WebPluginTypeEnabled"

@interface WebBasePluginPackage : NSObject
{
    NSMutableSet *pluginDatabases;
    
    WTF::String path;
    WebCore::PluginInfo pluginInfo;

    RetainPtr<CFBundleRef> cfBundle;

    BP_CreatePluginMIMETypesPreferencesFuncPtr BP_CreatePluginMIMETypesPreferences;
}

+ (WebBasePluginPackage *)pluginWithPath:(NSString *)pluginPath;
- (id)initWithPath:(NSString *)pluginPath;

- (BOOL)getPluginInfoFromPLists;

- (BOOL)load;
- (void)unload;

- (const WTF::String&)path;

- (const WebCore::PluginInfo&)pluginInfo;

- (String)bundleIdentifier;
- (String)bundleVersion;

- (BOOL)supportsExtension:(const WTF::String&)extension;
- (BOOL)supportsMIMEType:(const WTF::String&)MIMEType;

- (NSString *)MIMETypeForExtension:(const WTF::String&)extension;

- (BOOL)isQuickTimePlugIn;
- (BOOL)isJavaPlugIn;

- (BOOL)isNativeLibraryData:(NSData *)data;
- (UInt32)versionNumber;
- (void)wasAddedToPluginDatabase:(WebPluginDatabase *)database;
- (void)wasRemovedFromPluginDatabase:(WebPluginDatabase *)database;

@end
