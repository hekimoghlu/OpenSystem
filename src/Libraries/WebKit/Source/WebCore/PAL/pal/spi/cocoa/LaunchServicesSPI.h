/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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

#include <wtf/spi/darwin/XPCSPI.h>

#if USE(APPLE_INTERNAL_SDK)
#import <CoreServices/CoreServicesPriv.h>
#endif // USE(APPLE_INTERNAL_SDK)

#if HAVE(APP_LINKS)
@class LSAppLink;
typedef void (^LSAppLinkCompletionHandler)(LSAppLink *appLink, NSError *error);
typedef void (^LSAppLinkOpenCompletionHandler)(BOOL success, NSError *error);
#endif

#if USE(APPLE_INTERNAL_SDK)
// FIXME: remove the following section when <rdar://83360464> is fixed.
#if PLATFORM(MACCATALYST)
#if !defined(__LSAPPLICATIONSERVICESPRIV__)
enum LSSessionID {
    kLSDefaultSessionID = -2,
};
#endif // !defined(__LSAPPLICATIONSERVICESPRIV__)
WTF_EXTERN_C_BEGIN
CFDictionaryRef _LSApplicationCheckIn(LSSessionID, CFDictionaryRef applicationInfo);
WTF_EXTERN_C_END
#endif // PLATFORM(MACCATALYST)
#else // USE(APPLE_INTERNAL_SDK)

const uint8_t kLSOpenRunningInstanceBehaviorUseRunningProcess = 1;

@interface LSResourceProxy : NSObject <NSCopying, NSSecureCoding>
@property (nonatomic, copy, readonly) NSString *localizedName;
@end

@interface LSBundleProxy : LSResourceProxy <NSSecureCoding>
+ (LSBundleProxy *)bundleProxyWithAuditToken:(audit_token_t)auditToken error:(NSError **)outError;
@property (nonatomic, readonly) NSString *bundleIdentifier;
@end

#if HAVE(APP_LINKS)
@interface LSApplicationProxy : LSBundleProxy <NSSecureCoding>
@end

@interface LSAppLink : NSObject <NSSecureCoding>
@end

@interface _LSOpenConfiguration : NSObject <NSCopying, NSSecureCoding>
@property (readwrite) BOOL sensitive;
@property (readwrite) BOOL allowURLOverrides;
@property (readwrite, copy) NSDictionary<NSString *, id> *frontBoardOptions;
@property (readwrite, copy, nonatomic) NSURL *referrerURL;
@end

@interface LSAppLink ()
#if HAVE(APP_LINKS_WITH_ISENABLED)
+ (NSArray<LSAppLink *> *)appLinksWithURL:(NSURL *)aURL limit:(NSUInteger)limit error:(NSError **)outError;
- (void)openWithCompletionHandler:(LSAppLinkOpenCompletionHandler)completionHandler;
@property (nonatomic, getter=isEnabled) BOOL enabled;
#else
+ (void)getAppLinkWithURL:(NSURL *)aURL completionHandler:(LSAppLinkCompletionHandler)completionHandler;
- (void)openInWebBrowser:(BOOL)inWebBrowser setAppropriateOpenStrategyAndWebBrowserState:(NSDictionary<NSString *, id> *)state completionHandler:(LSAppLinkOpenCompletionHandler)completionHandler;
#endif // HAVE(APP_LINKS_WITH_ISENABLED)
+ (void)openWithURL:(NSURL *)aURL completionHandler:(LSAppLinkOpenCompletionHandler)completionHandler;
+ (void)openWithURL:(NSURL *)aURL configuration:(_LSOpenConfiguration *)configuration completionHandler:(LSAppLinkOpenCompletionHandler)completionHandler;
@property (readonly, strong) LSApplicationProxy *targetApplicationProxy;
@end
#endif // HAVE(APP_LINKS)

@interface NSURL ()
- (NSURL *)iTunesStoreURL;
@end

#if PLATFORM(MAC)
enum LSSessionID {
    kLSDefaultSessionID = -2,
};

enum {
    kLSServerConnectionStatusDoNotConnectToServerMask = 0x1ULL,
    kLSServerConnectionStatusReleaseNotificationsMask = (1ULL << 2),
};

WTF_EXTERN_C_BEGIN

CFDictionaryRef _LSApplicationCheckIn(LSSessionID, CFDictionaryRef applicationInfo);

WTF_EXTERN_C_END

#endif

#if HAVE(LSDATABASECONTEXT)
@interface LSDatabaseContext : NSObject
@property (class, readonly) LSDatabaseContext *sharedDatabaseContext;
@end
#endif

#endif // !USE(APPLE_INTERNAL_SDK)

#if HAVE(LSDATABASECONTEXT)
#if __has_include(<CoreServices/LSDatabaseContext+WebKit.h>)
#import <CoreServices/LSDatabaseContext+WebKit.h>
#elif !USE(APPLE_INTERNAL_SDK)
@interface LSDatabaseContext (WebKitChangeTracking)
- (id <NSObject>)addDatabaseChangeObserver4WebKit:(void (^)(xpc_object_t change))observer;
- (void)removeDatabaseChangeObserver4WebKit:(id <NSObject>)token;
- (void)observeDatabaseChange4WebKit:(xpc_object_t)change;

- (void)getSystemContentDatabaseObject4WebKit:(void (^)(xpc_object_t object, NSError *error))completion;
@end
#endif
#endif

#if PLATFORM(MAC)

typedef const struct CF_BRIDGED_TYPE(id) __LSASN* LSASNRef;
typedef enum LSSessionID LSSessionID;
typedef struct ProcessSerialNumber ProcessSerialNumber;

WTF_EXTERN_C_BEGIN

extern const CFStringRef _kLSAuditTokenKey;
extern const CFStringRef _kLSDisplayNameKey;
extern const CFStringRef _kLSOpenOptionActivateKey;
extern const CFStringRef _kLSOpenOptionAddToRecentsKey;
extern const CFStringRef _kLSOpenOptionBackgroundLaunchKey;
extern const CFStringRef _kLSOpenOptionHideKey;
extern const CFStringRef _kLSOpenOptionPreferRunningInstanceKey;
extern const CFStringRef _kLSPersistenceSuppressRelaunchAtLoginKey;

LSASNRef _LSGetCurrentApplicationASN();
LSASNRef _LSCopyLSASNForAuditToken(LSSessionID, audit_token_t);
OSStatus _LSSetApplicationInformationItem(LSSessionID, LSASNRef, CFStringRef keyToSetRef, CFTypeRef valueToSetRef, CFDictionaryRef* newInformationDictRef);
CFTypeRef _LSCopyApplicationInformationItem(LSSessionID, LSASNRef, CFTypeRef);
CFArrayRef _LSCopyMatchingApplicationsWithItems(LSSessionID, CFIndex count, CFTypeRef *keys, CFTypeRef *values);

typedef void (^ _LSOpenCompletionHandler)(LSASNRef, Boolean, CFErrorRef);
void _LSOpenURLsUsingBundleIdentifierWithCompletionHandler(CFArrayRef, CFStringRef, CFDictionaryRef, _LSOpenCompletionHandler);

WTF_EXTERN_C_END

#endif // PLATFORM(MAC)

#if PLATFORM(MAC) || PLATFORM(MACCATALYST)

WTF_EXTERN_C_BEGIN

typedef bool (^LSServerConnectionAllowedBlock) (CFDictionaryRef optionsRef);
void _LSSetApplicationLaunchServicesServerConnectionStatus(uint64_t flags, LSServerConnectionAllowedBlock block);

OSStatus _RegisterApplication(CFDictionaryRef, ProcessSerialNumber*);

WTF_EXTERN_C_END

#endif // PLATFORM(MAC) || PLATFORM(MACCATALYST)
