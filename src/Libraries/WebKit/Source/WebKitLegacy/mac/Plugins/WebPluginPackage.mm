/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#import <WebKitLegacy/WebPluginPackage.h>

#import <WebKitLegacy/WebKitLogging.h>
#import <WebKitLegacy/WebKitNSStringExtras.h>

NSString *WebPlugInBaseURLKey =                 @"WebPlugInBaseURLKey";
NSString *WebPlugInAttributesKey =              @"WebPlugInAttributesKey";
NSString *WebPlugInContainerKey =               @"WebPlugInContainerKey";
NSString *WebPlugInModeKey =                    @"WebPlugInModeKey";
NSString *WebPlugInShouldLoadMainResourceKey =  @"WebPlugInShouldLoadMainResourceKey";
NSString *WebPlugInContainingElementKey =       @"WebPlugInContainingElementKey";

@implementation WebPluginPackage

- (id)initWithPath:(NSString *)pluginPath
{
    if (!(self = [super initWithPath:pluginPath]))
        return nil;

    nsBundle = [[NSBundle alloc] initWithPath:path];

    if (!nsBundle) {
        [self release];
        return nil;
    }
    
    if (![[pluginPath pathExtension] _webkit_isCaseInsensitiveEqualToString:@"webplugin"]) {
        UInt32 type = 0;
        CFBundleGetPackageInfo(cfBundle.get(), &type, NULL);
        if (type != FOUR_CHAR_CODE('WBPL')) {
            [self release];
            return nil;
        }
    }
    
#if !PLATFORM(IOS_FAMILY)
    NSFileHandle *executableFile = [NSFileHandle fileHandleForReadingAtPath:[nsBundle executablePath]];
    NSData *data = [executableFile readDataOfLength:512];
    [executableFile closeFile];
    if (![self isNativeLibraryData:data]) {
        [self release];
        return nil;
    }
#endif

    if (![self getPluginInfoFromPLists]) {
        [self release];
        return nil;
    }

    return self;
}

- (void)dealloc
{
    [nsBundle release];

    [super dealloc];
}

- (Class)viewFactory
{
    return [nsBundle principalClass];
}

- (BOOL)load
{
#if !LOG_DISABLED
    CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
#endif
    
    // Load the bundle
    if (![nsBundle isLoaded]) {
        if (![nsBundle load])
            return NO;
    }
    
#if !LOG_DISABLED
    CFAbsoluteTime duration = CFAbsoluteTimeGetCurrent() - start;
    LOG(Plugins, "principalClass took %f seconds for: %@", duration, (NSString *)[self pluginInfo].name);
#endif
    return [super load];
}

- (NSBundle *)bundle
{
    return nsBundle;
}

@end

@implementation NSObject (WebScripting)

+ (BOOL)isSelectorExcludedFromWebScript:(SEL)aSelector
{
    return YES;
}

+ (BOOL)isKeyExcludedFromWebScript:(const char *)name
{
    return YES;
}

@end
