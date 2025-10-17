/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#import "config.h"
#import "_WKProcessPoolConfigurationInternal.h"

#import "LegacyGlobalSettings.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <objc/runtime.h>
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation _WKProcessPoolConfiguration

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<API::ProcessPoolConfiguration>(self);

    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKProcessPoolConfiguration.class, self))
        return;

    _processPoolConfiguration->~ProcessPoolConfiguration();

    [super dealloc];
}

- (NSURL *)injectedBundleURL
{
    return [NSURL fileURLWithPath:_processPoolConfiguration->injectedBundlePath()];
}

- (void)setInjectedBundleURL:(NSURL *)injectedBundleURL
{
    if (injectedBundleURL && !injectedBundleURL.isFileURL)
        [NSException raise:NSInvalidArgumentException format:@"Injected Bundle URL must be a file URL"];

    _processPoolConfiguration->setInjectedBundlePath(injectedBundleURL.path);
}

- (NSSet<Class> *)customClassesForParameterCoder
{
    return [NSSet set];
}

- (void)setCustomClassesForParameterCoder:(NSSet<Class> *)classesForCoder
{
}

- (NSUInteger)maximumProcessCount
{
    // Deprecated.
    return NSUIntegerMax;
}

- (void)setMaximumProcessCount:(NSUInteger)maximumProcessCount
{
    // Deprecated.
}

- (NSInteger)diskCacheSizeOverride
{
    return 0;
}

- (void)setDiskCacheSizeOverride:(NSInteger)size
{
}

- (BOOL)diskCacheSpeculativeValidationEnabled
{
    return NO;
}

- (void)setDiskCacheSpeculativeValidationEnabled:(BOOL)enabled
{
}

- (BOOL)ignoreSynchronousMessagingTimeoutsForTesting
{
    return _processPoolConfiguration->ignoreSynchronousMessagingTimeoutsForTesting();
}

- (void)setIgnoreSynchronousMessagingTimeoutsForTesting:(BOOL)ignoreSynchronousMessagingTimeoutsForTesting
{
    _processPoolConfiguration->setIgnoreSynchronousMessagingTimeoutsForTesting(ignoreSynchronousMessagingTimeoutsForTesting);
}

- (BOOL)attrStyleEnabled
{
    return _processPoolConfiguration->attrStyleEnabled();
}

- (void)setAttrStyleEnabled:(BOOL)enabled
{
    return _processPoolConfiguration->setAttrStyleEnabled(enabled);
}

- (BOOL)shouldThrowExceptionForGlobalConstantRedeclaration
{
    return _processPoolConfiguration->shouldThrowExceptionForGlobalConstantRedeclaration();
}

- (void)setShouldThrowExceptionForGlobalConstantRedeclaration:(BOOL)shouldThrow
{
    return _processPoolConfiguration->setShouldThrowExceptionForGlobalConstantRedeclaration(shouldThrow);
}

- (NSArray<NSURL *> *)additionalReadAccessAllowedURLs
{
    auto paths = _processPoolConfiguration->additionalReadAccessAllowedPaths();
    if (paths.isEmpty())
        return @[ ];

    return createNSArray(paths, [] (auto& path) {
        return [NSURL fileURLWithFileSystemRepresentation:path.utf8().data() isDirectory:NO relativeToURL:nil];
    }).autorelease();
}

- (void)setAdditionalReadAccessAllowedURLs:(NSArray<NSURL *> *)additionalReadAccessAllowedURLs
{
    Vector<String> paths;
    paths.reserveInitialCapacity(additionalReadAccessAllowedURLs.count);
    for (NSURL *url in additionalReadAccessAllowedURLs) {
        if (!url.isFileURL)
            [NSException raise:NSInvalidArgumentException format:@"%@ is not a file URL", url];

        paths.append(String::fromUTF8(url.fileSystemRepresentation));
    }

    _processPoolConfiguration->setAdditionalReadAccessAllowedPaths(WTFMove(paths));
}

#if PLATFORM(IOS_FAMILY) && !PLATFORM(IOS_FAMILY_SIMULATOR)
- (NSUInteger)wirelessContextIdentifier
{
    return 0;
}

- (void)setWirelessContextIdentifier:(NSUInteger)identifier
{
}
#endif

- (NSArray *)cachePartitionedURLSchemes
{
    return createNSArray(_processPoolConfiguration->cachePartitionedURLSchemes()).autorelease();
}

- (void)setCachePartitionedURLSchemes:(NSArray *)cachePartitionedURLSchemes
{
    _processPoolConfiguration->setCachePartitionedURLSchemes(makeVector<String>(cachePartitionedURLSchemes));
}

- (NSArray *)alwaysRevalidatedURLSchemes
{
    return createNSArray(_processPoolConfiguration->alwaysRevalidatedURLSchemes()).autorelease();
}

- (void)setAlwaysRevalidatedURLSchemes:(NSArray *)alwaysRevalidatedURLSchemes
{
    _processPoolConfiguration->setAlwaysRevalidatedURLSchemes(makeVector<String>(alwaysRevalidatedURLSchemes));
}

- (NSString *)sourceApplicationBundleIdentifier
{
    return nil;
}

- (void)setSourceApplicationBundleIdentifier:(NSString *)sourceApplicationBundleIdentifier
{
}

- (NSString *)sourceApplicationSecondaryIdentifier
{
    return nil;
}

- (void)setSourceApplicationSecondaryIdentifier:(NSString *)sourceApplicationSecondaryIdentifier
{
}

- (void)setPresentingApplicationPID:(pid_t)presentingApplicationPID
{
    _processPoolConfiguration->setPresentingApplicationPID(presentingApplicationPID);
}

- (pid_t)presentingApplicationPID
{
    return _processPoolConfiguration->presentingApplicationPID();
}

- (void)setPresentingApplicationProcessToken:(audit_token_t)token
{
    _processPoolConfiguration->setPresentingApplicationProcessToken(token);
}

- (audit_token_t)presentingApplicationProcessToken
{
    if (_processPoolConfiguration->presentingApplicationProcessToken())
        return *_processPoolConfiguration->presentingApplicationProcessToken();
    return { };
}

- (void)setProcessSwapsOnNavigation:(BOOL)swaps
{
    _processPoolConfiguration->setProcessSwapsOnNavigation(swaps);
}

- (BOOL)processSwapsOnNavigation
{
    return _processPoolConfiguration->processSwapsOnNavigation();
}

- (void)setPrewarmsProcessesAutomatically:(BOOL)prewarms
{
    _processPoolConfiguration->setIsAutomaticProcessWarmingEnabled(prewarms);
}

- (BOOL)prewarmsProcessesAutomatically
{
    return _processPoolConfiguration->isAutomaticProcessWarmingEnabled();
}

- (void)setUsesWebProcessCache:(BOOL)value
{
    _processPoolConfiguration->setUsesWebProcessCache(value);
}

- (BOOL)usesWebProcessCache
{
    return _processPoolConfiguration->usesWebProcessCache();
}

- (void)setAlwaysKeepAndReuseSwappedProcesses:(BOOL)swaps
{
    _processPoolConfiguration->setAlwaysKeepAndReuseSwappedProcesses(swaps);
}

- (BOOL)alwaysKeepAndReuseSwappedProcesses
{
    return _processPoolConfiguration->alwaysKeepAndReuseSwappedProcesses();
}

- (void)setProcessSwapsOnNavigationWithinSameNonHTTPFamilyProtocol:(BOOL)swaps
{
    _processPoolConfiguration->setProcessSwapsOnNavigationWithinSameNonHTTPFamilyProtocol(swaps);
}

- (BOOL)processSwapsOnNavigationWithinSameNonHTTPFamilyProtocol
{
    return _processPoolConfiguration->processSwapsOnNavigationWithinSameNonHTTPFamilyProtocol();
}

- (BOOL)pageCacheEnabled
{
    return _processPoolConfiguration->usesBackForwardCache();
}

- (void)setPageCacheEnabled:(BOOL)enabled
{
    return _processPoolConfiguration->setUsesBackForwardCache(enabled);
}

- (BOOL)usesSingleWebProcess
{
    return _processPoolConfiguration->usesSingleWebProcess();
}

- (void)setUsesSingleWebProcess:(BOOL)enabled
{
    _processPoolConfiguration->setUsesSingleWebProcess(enabled);
}

- (BOOL)isJITEnabled
{
    return _processPoolConfiguration->isJITEnabled();
}

- (void)setJITEnabled:(BOOL)enabled
{
    _processPoolConfiguration->setJITEnabled(enabled);
}

#if PLATFORM(IOS_FAMILY)
- (BOOL)alwaysRunsAtBackgroundPriority
{
    return _processPoolConfiguration->alwaysRunsAtBackgroundPriority();
}

- (void)setAlwaysRunsAtBackgroundPriority:(BOOL)alwaysRunsAtBackgroundPriority
{
    _processPoolConfiguration->setAlwaysRunsAtBackgroundPriority(alwaysRunsAtBackgroundPriority);
}

- (BOOL)shouldTakeUIBackgroundAssertion
{
    return _processPoolConfiguration->shouldTakeUIBackgroundAssertion();
}

- (void)setShouldTakeUIBackgroundAssertion:(BOOL)shouldTakeUIBackgroundAssertion
{
    return _processPoolConfiguration->setShouldTakeUIBackgroundAssertion(shouldTakeUIBackgroundAssertion);
}
#endif

- (NSString *)description
{
    NSString *description = [NSString stringWithFormat:@"<%@: %p", NSStringFromClass(self.class), self];

    if (!_processPoolConfiguration->injectedBundlePath().isEmpty())
        return [description stringByAppendingFormat:@"; injectedBundleURL: \"%@\">", [self injectedBundleURL]];

    return [description stringByAppendingString:@">"];
}

- (id)copyWithZone:(NSZone *)zone
{
    return [wrapper(_processPoolConfiguration->copy()) retain];
}

- (NSString *)customWebContentServiceBundleIdentifier
{
    return nil;
}

- (void)setCustomWebContentServiceBundleIdentifier:(NSString *)customWebContentServiceBundleIdentifier
{
}

- (BOOL)configureJSCForTesting
{
    return _processPoolConfiguration->shouldConfigureJSCForTesting();
}

- (void)setConfigureJSCForTesting:(BOOL)value
{
    _processPoolConfiguration->setShouldConfigureJSCForTesting(value);
}

- (NSString *)timeZoneOverride
{
    return _processPoolConfiguration->timeZoneOverride();
}

- (void)setTimeZoneOverride:(NSString *)timeZone
{
    _processPoolConfiguration->setTimeZoneOverride(timeZone);
}

- (void)setMemoryFootprintPollIntervalForTesting:(NSTimeInterval)pollInterval
{
    _processPoolConfiguration->setMemoryFootprintPollIntervalForTesting(Seconds { pollInterval });
}

- (NSTimeInterval)memoryFootprintPollIntervalForTesting
{
    return _processPoolConfiguration->memoryFootprintPollIntervalForTesting().seconds();
}

- (NSArray<NSNumber *> *)memoryFootprintNotificationThresholds
{
    const auto& thresholds = _processPoolConfiguration->memoryFootprintNotificationThresholds();
    RetainPtr result = adoptNS([[NSMutableArray alloc] initWithCapacity: thresholds.size()]);
    for (auto& threshold : thresholds)
        [result addObject:@(threshold)];
    return result.autorelease();
}

- (void)setMemoryFootprintNotificationThresholds:(NSArray<NSNumber *> *)thresholds
{
    Vector<size_t> sizes;
    sizes.reserveCapacity(thresholds.count);
    for (NSNumber *threshold in thresholds)
        sizes.append(static_cast<size_t>(threshold.unsignedLongLongValue));
    _processPoolConfiguration->setMemoryFootprintNotificationThresholds(WTFMove(sizes));
}

- (BOOL)suspendsWebProcessesAggressivelyOnMemoryPressure
{
#if ENABLE(WEB_PROCESS_SUSPENSION_DELAY)
    return _processPoolConfiguration->suspendsWebProcessesAggressivelyOnMemoryPressure();
#else
    return NO;
#endif
}

- (void)setSuspendsWebProcessesAggressivelyOnMemoryPressure:(BOOL)enabled
{
#if ENABLE(WEB_PROCESS_SUSPENSION_DELAY)
    _processPoolConfiguration->setSuspendsWebProcessesAggressivelyOnMemoryPressure(enabled);
#endif
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_processPoolConfiguration;
}

@end
