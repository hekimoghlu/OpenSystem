/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
#import "WKFullKeyboardAccessWatcher.h"

#if ENABLE(FULL_KEYBOARD_ACCESS)

#import "WebProcessPool.h"

#if PLATFORM(IOS_FAMILY)
#import "AccessibilitySupportSPI.h"
#endif

#if PLATFORM(MAC)
static NSString * const KeyboardUIModeDidChangeNotification = @"com.apple.KeyboardUIModeDidChange";
static const CFStringRef AppleKeyboardUIMode = CFSTR("AppleKeyboardUIMode");
#endif

@implementation WKFullKeyboardAccessWatcher

static inline BOOL platformIsFullKeyboardAccessEnabled()
{
    BOOL fullKeyboardAccessEnabled = NO;
#if PLATFORM(MAC)
    CFPreferencesAppSynchronize(kCFPreferencesCurrentApplication);
    
    Boolean keyExistsAndHasValidFormat;
    int mode = CFPreferencesGetAppIntegerValue(AppleKeyboardUIMode, kCFPreferencesCurrentApplication, &keyExistsAndHasValidFormat);
    if (keyExistsAndHasValidFormat) {
        // The keyboard access mode has two bits:
        // Bit 0 is set if user can set the focus to menus, the dock, and various windows using the keyboard.
        // Bit 1 is set if controls other than text fields are included in the tab order (WebKit also always includes lists).
        fullKeyboardAccessEnabled = (mode & 0x2);
    }
#elif PLATFORM(IOS_FAMILY)
    fullKeyboardAccessEnabled = _AXSFullKeyboardAccessEnabled();
#endif
    
    return fullKeyboardAccessEnabled;
}

- (void)notifyAllProcessPools
{
    for (auto& processPool : WebKit::WebProcessPool::allProcessPools())
        processPool->fullKeyboardAccessModeChanged(fullKeyboardAccessEnabled);
}

- (void)retrieveKeyboardUIModeFromPreferences:(NSNotification *)notification
{
    BOOL oldValue = fullKeyboardAccessEnabled;

    fullKeyboardAccessEnabled = platformIsFullKeyboardAccessEnabled();
    
    if (fullKeyboardAccessEnabled != oldValue)
        [self notifyAllProcessPools];
}

- (id)init
{
    self = [super init];
    if (!self)
        return nil;

    [self retrieveKeyboardUIModeFromPreferences:nil];

    NSNotificationCenter *notificationCenter = nil;
    NSString *notitificationName = nil;
    
#if PLATFORM(MAC)
    notificationCenter = [NSDistributedNotificationCenter defaultCenter];
    notitificationName = KeyboardUIModeDidChangeNotification;
#elif PLATFORM(IOS_FAMILY)
    notificationCenter = [NSNotificationCenter defaultCenter];
    notitificationName = (NSString *)kAXSFullKeyboardAccessEnabledNotification;
#endif
    
    if (notitificationName)
        [notificationCenter addObserver:self selector:@selector(retrieveKeyboardUIModeFromPreferences:) name:notitificationName object:nil];

    return self;
}

+ (BOOL)fullKeyboardAccessEnabled
{
    static WKFullKeyboardAccessWatcher *watcher = [[WKFullKeyboardAccessWatcher alloc] init];
    return watcher->fullKeyboardAccessEnabled;
}

@end

#endif // ENABLE(FULL_KEYBOARD_ACCESS)
