/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
#if OCTAGON

#if __has_include(<UserManagement/UserManagement.h>)
#import <UserManagement/UserManagement.h>
#endif

#import "keychain/ot/OTPersonaAdapter.h"
#import "ipc/securityd_client.h"
#import "utilities/debugging.h"

@implementation OTPersonaActualAdapter

- (instancetype)init
{
    if((self = [super init])) {
    }
    return self;
}

- (NSString* _Nullable)currentThreadPersonaUniqueString
{
#if KEYCHAIN_SUPPORTS_PERSONA_MULTIUSER
    UMUserPersona * persona = [[UMUserManager sharedManager] currentPersona];
    return persona.userPersonaUniqueString;
#else
    return nil;
#endif  // KEYCHAIN_SUPPORTS_PERSONA_MULTIUSER
}

- (BOOL)currentThreadIsForPrimaryiCloudAccount
{
#if KEYCHAIN_SUPPORTS_PERSONA_MULTIUSER
    UMUserPersona * persona = [[UMUserManager sharedManager] currentPersona];

    switch(persona.userPersonaType) {
            /*
             * Apps launch as Personal, Guest, or Enterprise.
             *
             * Daemons launch in either Default or System, and can become Personal/Guest/Enterprise while handling an XPC,
             * or (if they are System) they can adopt a Personal/Guest/Enterprise Persona at runtime. So, if the incoming XPC is
             * not Guest/Enterprise, the XPC is in the context of the primary iCloud account.
             *
             * Invalid can happen on macOS for non-app-store binaries, and still should be considered as for the primary 'user'.
             *
             * Universal shouldn't ever be seen at runtime, and Managed is deprecated and should be unused.
             */
        case UMUserPersonaTypeDefault:
        case UMUserPersonaTypeSystem:
        case UMUserPersonaTypePersonal:
        case UMUserPersonaTypeInvalid:
            return YES;
            /*
             * Guests and enterprise accounts are not for the primary account.
             */
        case UMUserPersonaTypeGuest:
        case UMUserPersonaTypeEnterprise:
            return NO;

        case UMUserPersonaTypeUniversal:
        case UMUserPersonaTypeManaged:
        default:
            secnotice("persona", "Received unexpected Universal/Managed/other persona; treating as not for primary account: %@(%d)",
                      persona.userPersonaUniqueString,
                      (int)persona.userPersonaType);
            return NO;
    }

#else
    return YES;
#endif
}


- (void)prepareThreadForKeychainAPIUseForPersonaIdentifier:(NSString* _Nullable)personaUniqueString
{
#if KEYCHAIN_SUPPORTS_PERSONA_MULTIUSER
    NSError* error = [[UMUserPersona currentPersona] generateAndRestorePersonaContextWithPersonaUniqueString:personaUniqueString];

    if(error != nil) {
        secnotice("ckks-persona", "Unable to adopt persona %@: %@", personaUniqueString, error);
    } else {
        secinfo("ckks-persona", "Adopted persona for id '%@'", personaUniqueString);
    }
#endif
}


- (void)performBlockWithPersonaIdentifier:(NSString* _Nullable)personaUniqueString
                                     block:(void (^) (void)) block
{
#if KEYCHAIN_SUPPORTS_PERSONA_MULTIUSER
    NSString* oldPersonaString = [self currentThreadPersonaUniqueString];
    if([personaUniqueString isEqualToString: oldPersonaString]) {
        block();
        return;
    }
    
    [self prepareThreadForKeychainAPIUseForPersonaIdentifier: personaUniqueString];
    block();
    [self prepareThreadForKeychainAPIUseForPersonaIdentifier: oldPersonaString];
    
#else
    block();
#endif
}


@end

#endif // Octagon
