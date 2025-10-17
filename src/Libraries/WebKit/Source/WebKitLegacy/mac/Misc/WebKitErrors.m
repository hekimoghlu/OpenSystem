/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#import <WebKitLegacy/WebKitErrors.h>

#import "WebLocalizableStringsInternal.h"
#import <Foundation/NSURLError.h>
#import <WebKitLegacy/WebKitErrorsPrivate.h>
#import <WebKitLegacy/WebNSURLExtras.h>

#import <dispatch/dispatch.h>

NSString *WebKitErrorDomain = @"WebKitErrorDomain";

NSString * const WebKitErrorMIMETypeKey =               @"WebKitErrorMIMETypeKey";
NSString * const WebKitErrorPlugInNameKey =             @"WebKitErrorPlugInNameKey";
NSString * const WebKitErrorPlugInPageURLStringKey =    @"WebKitErrorPlugInPageURLStringKey";

// Policy errors
#define WebKitErrorDescriptionCannotShowMIMEType UI_STRING_INTERNAL("Content with specified MIME type canâ€™t be shown", "WebKitErrorCannotShowMIMEType description")
#define WebKitErrorDescriptionCannotShowURL UI_STRING_INTERNAL("The URL canâ€™t be shown", "WebKitErrorCannotShowURL description")
#define WebKitErrorDescriptionFrameLoadInterruptedByPolicyChange UI_STRING_INTERNAL("Frame load interrupted", "WebKitErrorFrameLoadInterruptedByPolicyChange description")
#define WebKitErrorDescriptionCannotUseRestrictedPort UI_STRING_INTERNAL("Not allowed to use restricted network port", "WebKitErrorCannotUseRestrictedPort description")
#define WebKitErrorDescriptionFrameLoadBlockedByContentFilter UI_STRING_INTERNAL("The URL was blocked by a content filter", "WebKitErrorFrameLoadBlockedByContentFilter description")

// Plug-in and java errors
#define WebKitErrorDescriptionCannotFindPlugin UI_STRING_INTERNAL("The plug-in canâ€™t be found", "WebKitErrorCannotFindPlugin description")
#define WebKitErrorDescriptionCannotLoadPlugin UI_STRING_INTERNAL("The plug-in canâ€™t be loaded", "WebKitErrorCannotLoadPlugin description")
#define WebKitErrorDescriptionJavaUnavailable UI_STRING_INTERNAL("Java is unavailable", "WebKitErrorJavaUnavailable description")
#define WebKitErrorDescriptionPlugInCancelledConnection UI_STRING_INTERNAL("Plug-in cancelled", "WebKitErrorPlugInCancelledConnection description")
#define WebKitErrorDescriptionPlugInWillHandleLoad UI_STRING_INTERNAL("Plug-in handled load", "WebKitErrorPlugInWillHandleLoad description")

// Geolocations errors

#define WebKitErrorDescriptionGeolocationLocationUnknown UI_STRING_INTERNAL("The current location cannot be found.", "WebKitErrorGeolocationLocationUnknown description")

static NSMutableDictionary *descriptions = nil;

@interface NSError (WebKitInternal)
- (instancetype)_webkit_initWithDomain:(NSString *)domain code:(int)code URL:(NSURL *)URL __attribute__((objc_method_family(init)));
@end

@implementation NSError (WebKitInternal)

- (instancetype)_webkit_initWithDomain:(NSString *)domain code:(int)code URL:(NSURL *)URL
{
    // Insert a localized string here for those folks not savvy to our category methods.
    NSString *localizedDescription = [[descriptions objectForKey:domain] objectForKey:@(code)];
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    NSDictionary *userInfo = [NSDictionary dictionaryWithObjectsAndKeys:
        URL, @"NSErrorFailingURLKey",
        [URL absoluteString], NSURLErrorFailingURLStringErrorKey,
        localizedDescription, NSLocalizedDescriptionKey,
        nil];
    ALLOW_DEPRECATED_DECLARATIONS_END
    return [self initWithDomain:domain code:code userInfo:userInfo];
}

@end

@implementation NSError (WebKitExtras)

+ (void)_registerWebKitErrors
{
    static dispatch_once_t flag;
    dispatch_once(&flag, ^{
        @autoreleasepool {
            NSDictionary *dict = [NSDictionary dictionaryWithObjectsAndKeys:
                // Policy errors
                WebKitErrorDescriptionCannotShowMIMEType,                   @(WebKitErrorCannotShowMIMEType),
                WebKitErrorDescriptionCannotShowURL,                        @(WebKitErrorCannotShowURL),
                WebKitErrorDescriptionFrameLoadInterruptedByPolicyChange,   @(WebKitErrorFrameLoadInterruptedByPolicyChange),
                WebKitErrorDescriptionCannotUseRestrictedPort,              @(WebKitErrorCannotUseRestrictedPort),
                WebKitErrorDescriptionFrameLoadBlockedByContentFilter,      @(WebKitErrorFrameLoadBlockedByContentFilter),

                // Plug-in and java errors
                WebKitErrorDescriptionCannotFindPlugin,                     @(WebKitErrorCannotFindPlugIn),
                WebKitErrorDescriptionCannotLoadPlugin,                     @(WebKitErrorCannotLoadPlugIn),
                WebKitErrorDescriptionJavaUnavailable,                      @(WebKitErrorJavaUnavailable),
                WebKitErrorDescriptionPlugInCancelledConnection,            @(WebKitErrorPlugInCancelledConnection),
                WebKitErrorDescriptionPlugInWillHandleLoad,                 @(WebKitErrorPlugInWillHandleLoad),

                // Geolocation errors
                WebKitErrorDescriptionGeolocationLocationUnknown,           @(WebKitErrorGeolocationLocationUnknown),
                nil];

            [NSError _webkit_addErrorsWithCodesAndDescriptions:dict inDomain:WebKitErrorDomain];
        }
    });
}

+(id)_webkit_errorWithDomain:(NSString *)domain code:(int)code URL:(NSURL *)URL
{
    return [[[self alloc] _webkit_initWithDomain:domain code:code URL:URL] autorelease];
}

+ (NSError *)_webKitErrorWithDomain:(NSString *)domain code:(int)code URL:(NSURL *)URL
{
    [self _registerWebKitErrors];
    return [self _webkit_errorWithDomain:domain code:code URL:URL];
}

+ (NSError *)_webKitErrorWithCode:(int)code failingURL:(NSString *)URLString
{
    return [self _webKitErrorWithDomain:WebKitErrorDomain code:code URL:[NSURL _webkit_URLWithUserTypedString:URLString]];
}

- (id)_initWithPluginErrorCode:(int)code
                    contentURL:(NSURL *)contentURL
                 pluginPageURL:(NSURL *)pluginPageURL
                    pluginName:(NSString *)pluginName
                      MIMEType:(NSString *)MIMEType
{
    [[self class] _registerWebKitErrors];
    
    NSMutableDictionary *userInfo = [[NSMutableDictionary alloc] init];
    NSDictionary *descriptionsForWebKitErrorDomain = [descriptions objectForKey:WebKitErrorDomain];
    NSString *localizedDescription = [descriptionsForWebKitErrorDomain objectForKey:@(code)];
    if (localizedDescription)
        [userInfo setObject:localizedDescription forKey:NSLocalizedDescriptionKey];
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    if (contentURL) {
        [userInfo setObject:contentURL forKey:@"NSErrorFailingURLKey"];
        [userInfo setObject:[contentURL _web_userVisibleString] forKey:NSURLErrorFailingURLStringErrorKey];
    }
    ALLOW_DEPRECATED_DECLARATIONS_END
    if (pluginPageURL) {
        [userInfo setObject:[pluginPageURL _web_userVisibleString] forKey:WebKitErrorPlugInPageURLStringKey];
    }
    if (pluginName) {
        [userInfo setObject:pluginName forKey:WebKitErrorPlugInNameKey];
    }
    if (MIMEType) {
        [userInfo setObject:MIMEType forKey:WebKitErrorMIMETypeKey];
    }

    NSDictionary *userInfoCopy = [userInfo count] > 0 ? [[NSDictionary alloc] initWithDictionary:userInfo] : nil;
    [userInfo release];
    NSError *error = [self initWithDomain:WebKitErrorDomain code:code userInfo:userInfoCopy];
    [userInfoCopy release];
    
    return error;
}

+ (void)_webkit_addErrorsWithCodesAndDescriptions:(NSDictionary *)dictionary inDomain:(NSString *)domain
{
    if (!descriptions)
        descriptions = [[NSMutableDictionary alloc] init];

    [descriptions setObject:dictionary forKey:domain];
}

@end
