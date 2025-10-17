/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
#import "ManagedAppManager.h"

#import <SoftLinking/SoftLinking.h>

#if TARGET_OS_IOS
#import <ManagedConfiguration/ManagedConfiguration.h>
SOFT_LINK_FRAMEWORK(PrivateFrameworks, ManagedConfiguration)
SOFT_LINK_CLASS(ManagedConfiguration, MCProfileConnection)
SOFT_LINK_CONSTANT(ManagedConfiguration, MCManagedAppsChangedNotification, NSString *)
#elif TARGET_OS_OSX
#import <ConfigurationProfiles/ConfigurationProfiles.h>
#import <libproc.h>
SOFT_LINK_FRAMEWORK(PrivateFrameworks, ConfigurationProfiles)
SOFT_LINK_FUNCTION(ConfigurationProfiles, CP_ManagedAppsIsAppManagedAtURL, __CP_ManagedAppsIsAppManagedAtURL, BOOL, (NSURL* appURL, NSString* bundleID), (appURL, bundleID));
#endif

@interface ManagedAppManager ()

@property (nonatomic) NSArray<NSString *> *managedApps;

@end

@implementation ManagedAppManager
@synthesize managedApps;

- (instancetype)init
{
    self = [super init];
    if (self) {
	managedApps = @[];
    }
    return self;
}

- (void)dealloc
{
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (BOOL)isManagedApp:(NSString*)bundleId auditToken:(audit_token_t)auditToken
{
#if TARGET_OS_IOS
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
	[self listenForChanges];
	[self updateManagedApps:nil];
    });
    
    @synchronized (self) {
	return [self.managedApps containsObject:bundleId];
    }
#elif TARGET_OS_OSX
    NSURL *appURL = nil;
    char buffer[PROC_PIDPATHINFO_MAXSIZE] = {};
    int size = 0;
    size = proc_pidpath_audittoken(&auditToken, buffer, sizeof(buffer));
    if (size <= 0) {
	os_log_error(GSSOSLog(), "isManagedApp: proc_pidpath_audittoken failed for %{public}@", @(audit_token_to_pid(auditToken)));
	return false;
    }
    NSString *path = [NSString stringWithCString:buffer encoding:NSUTF8StringEncoding];
    os_log_debug(GSSOSLog(), "isManagedApp: %{public}@: %{public}@", @(audit_token_to_pid(auditToken)), path);
    if (!path) {
	os_log_error(GSSOSLog(), "isManagedApp: path not found for %{public}@", @(audit_token_to_pid(auditToken)));
	return false;
    }

    appURL = CFBridgingRelease(CFURLCreateWithFileSystemPath(kCFAllocatorDefault, (__bridge CFStringRef)path, kCFURLPOSIXPathStyle, FALSE));
    os_log_debug(GSSOSLog(), "isManagedApp: CFURLCreateWithFileSystemPath: %{public}@", appURL);


    BOOL managed = appURL ? __CP_ManagedAppsIsAppManagedAtURL(appURL, bundleId) : NO;
    return managed;
#else
    return false;
#endif
}

#if TARGET_OS_IOS
- (void)updateManagedApps:(NSNotification *)notification
{
    os_log_info(GSSOSLog(), "Updating Managed App list");
    os_log_debug(GSSOSLog(), "Old Managed App list: %{private}@", self.managedApps);
    @synchronized (self) {
	self.managedApps = [[getMCProfileConnectionClass() sharedConnection] managedAppBundleIDs];
    }
    os_log_debug(GSSOSLog(), "New Managed App list: %{private}@", self.managedApps);
}

- (void)listenForChanges
{
    NSString *notificationName = getMCManagedAppsChangedNotification();
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(updateManagedApps:) name:notificationName object:[getMCProfileConnectionClass() sharedConnection]];
}
#endif

@end
