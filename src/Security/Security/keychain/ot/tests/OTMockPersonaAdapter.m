/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
#import <TargetConditionals.h>
#import "ipc/securityd_client.h"
#import "keychain/ot/tests/OTMockPersonaAdapter.h"

// This mock layer will fail for persona OctagonTests once we adopt UserManagement test infrastructure (when it becomes available)

@implementation OTMockPersonaAdapter

- (instancetype)init
{
    if((self = [super init])) {
        _isDefaultPersona = YES;
        _currentPersonaString = [OTMockPersonaAdapter defaultMockPersonaString];
    }
    return self;
}

- (NSString*)currentThreadPersonaUniqueString
{
    return self.currentPersonaString;
}

- (BOOL)currentThreadIsForPrimaryiCloudAccount {
    return self.isDefaultPersona;
}

+ (NSString*)defaultMockPersonaString
{
    return @"MOCK_PERSONA_IDENTIFIER";
}

- (void)prepareThreadForKeychainAPIUseForPersonaIdentifier:(NSString* _Nullable)personaUniqueString
{
    
#if KEYCHAIN_SUPPORTS_PERSONA_MULTIUSER
    // Note that this is a global override, and so is not thread-safe at all.
    // I can't find a way to simulate persona attachment to threads in the face of dispatch_async.
    // If you get strange test behavior with the keychain, suspect simultaneous access from different threads with expected persona musrs.
    if(personaUniqueString == nil || [personaUniqueString isEqualToString:[OTMockPersonaAdapter defaultMockPersonaString]]) {
        SecSecuritySetPersonaMusr(NULL);
    } else {
        SecSecuritySetPersonaMusr((__bridge CFStringRef)personaUniqueString);
    }
#endif
}

- (void)performBlockWithPersonaIdentifier:(NSString* _Nullable)personaUniqueString
                                     block:(void (^) (void)) block
{
#if KEYCHAIN_SUPPORTS_PERSONA_MULTIUSER
    [self prepareThreadForKeychainAPIUseForPersonaIdentifier: personaUniqueString];
    block();
    // once UserManagement supplies some testing infrastructure that actually changes the current thread's persona in xctests, we should change this routine to
    // mimick the performBlockWithPersonaIdentifier from OTPersonaAdapter (where we look up the current thread's persona and restore it after the block executes.)
    [self prepareThreadForKeychainAPIUseForPersonaIdentifier: personaUniqueString];
#else
    block();
#endif
}


@end
