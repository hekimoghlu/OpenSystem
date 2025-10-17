/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
#import "SecC2DeviceInfo.h"

#import <os/variant_private.h>
#import <CoreFoundation/CFPriv.h>
#import <os/log.h>
#if TARGET_OS_IPHONE
#import <MobileGestalt.h>
#else
#import <sys/sysctl.h>
#endif

static NSString* C2MetricBuildVersion(void);
static NSString* C2MetricProductName(void);
static NSString* C2MetricProductType(void);
static NSString* C2MetricProductVersion(void);
static NSString* C2MetricProcessName(void);
static NSString* C2MetricProcessVersion(void);
static NSString* C2MetricProcessUUID(void);

@implementation SecC2DeviceInfo

+ (BOOL) isAppleInternal {
    return os_variant_has_internal_content("com.apple.security.analytics");
}

+ (NSString*) buildVersion {
    return C2MetricBuildVersion();
}

+ (NSString*) productName {
    return C2MetricProductName();
}

+ (NSString*) productType {
    return C2MetricProductType();
}

+ (NSString*) productVersion {
    return C2MetricProductVersion();
}

+ (NSString*) processName {
    return C2MetricProcessName();
}

+ (NSString*) processVersion {
    return C2MetricProcessVersion();
}

+ (NSString*) processUUID {
    return C2MetricProcessUUID();
}

@end

/* Stolen without remorse from CloudKit. */

#pragma mark - NSBundleInfoDictionary Constants

static NSDictionary *processInfoDict(void) {
    static NSDictionary *processInfoDict = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSBundle *processBundle = [NSBundle mainBundle];
        processInfoDict = [processBundle infoDictionary];
    });
    return processInfoDict;
}

static NSString* C2MetricProcessName(void) {
    return processInfoDict()[(NSString *)kCFBundleIdentifierKey];
}

static NSString* C2MetricProcessVersion(void) {
    return processInfoDict()[(NSString *)_kCFBundleShortVersionStringKey];
}

static NSString* C2MetricProcessUUID(void) {
    static NSString* processUUIDString;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        processUUIDString = [[NSUUID UUID] UUIDString];
    });
    return processUUIDString;
}

#pragma mark - MobileGestalt Constants

#if TARGET_OS_IPHONE

static NSMutableDictionary* _CKCachedGestaltValues = nil;

static NSArray* _CKCachedLockdownKeys(void) {
    return @[(NSString *)kMGQUniqueDeviceID,
             (NSString *)kMGQBuildVersion,
             (NSString *)kMGQProductName,
             (NSString *)kMGQProductType,
             (NSString *)kMGQProductVersion];
}

static NSDictionary* _CKGetCachedGestaltValues(void) {
    static dispatch_once_t pred;
    dispatch_once(&pred, ^{
        _CKCachedGestaltValues = [[NSMutableDictionary alloc] initWithCapacity:0];

        for (NSString *key in _CKCachedLockdownKeys()) {
            NSString *value = CFBridgingRelease(MGCopyAnswer((__bridge CFStringRef)key, NULL));
            if (value) {
                _CKCachedGestaltValues[key] = value;
            } else {
                os_log(OS_LOG_DEFAULT, "Error getting %@ from MobileGestalt", key);
            }
        }
    });
    return _CKCachedGestaltValues;
}

static NSString* _CKGetCachedGestaltValue(NSString *key) {
    return _CKGetCachedGestaltValues()[key];
}

static NSString* C2MetricBuildVersion(void) {
    return _CKGetCachedGestaltValue((NSString *)kMGQBuildVersion);
}

static NSString* C2MetricProductName(void) {
    return _CKGetCachedGestaltValue((NSString *)kMGQProductName);
}

static NSString* C2MetricProductType(void) {
    return _CKGetCachedGestaltValue((NSString *)kMGQProductType);
}

static NSString* C2MetricProductVersion(void) {
    return _CKGetCachedGestaltValue((NSString *)kMGQProductVersion);
}

#else

static CFStringRef CKCopySysctl(int mib[2]) {
    char sysctlString[128];
    size_t len = sizeof(sysctlString);

    // add the system product
    if (sysctl(mib, 2, sysctlString, &len, 0, 0) >= 0) {
        return CFStringCreateWithCString(kCFAllocatorDefault, sysctlString, kCFStringEncodingUTF8);
    }

    return NULL;
}

static NSDictionary* systemVersionDict(void) {
    static NSDictionary *sysVers = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sysVers = (__bridge NSDictionary *)_CFCopySystemVersionDictionary();
    });
    return sysVers;
}

static NSString* C2MetricBuildVersion(void) {
    return systemVersionDict()[(NSString *)_kCFSystemVersionBuildVersionKey];
}

static NSString* C2MetricProductName(void) {
    return systemVersionDict()[(NSString *)_kCFSystemVersionProductNameKey];
}

static NSString* C2MetricProductType(void) {
    static dispatch_once_t onceToken;
    static NSString *productType = nil;
    dispatch_once(&onceToken, ^{
        productType = (__bridge NSString *)CKCopySysctl((int[2]) { CTL_HW, HW_MODEL });
    });
    return productType;
}

static NSString* C2MetricProductVersion(void) {
    return systemVersionDict()[(NSString *)_kCFSystemVersionProductVersionKey];
}

#endif
