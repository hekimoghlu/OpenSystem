/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#import "KeychainDataclassOwner.h"
#import "NSError+UsefulConstructors.h"
#import "SecCFRelease.h"
#import "SOSCloudCircle.h"
#import "debugging.h"
#import <Accounts/Accounts.h>
#import <Accounts/ACDataclassAction.h>
#import <Accounts/ACConstants.h>
#import <Security/Security.h>
#import <Security/SecItemPriv.h>

@implementation KeychainDataclassOwner


+ (NSArray*)dataclasses
{
    return @[kAccountDataclassKeychainSync];
}

- (NSArray*)actionsForDeletingAccount:(ACAccount*)account forDataclass:(NSString*)dataclass
{
    if (![dataclass isEqual:kAccountDataclassKeychainSync]) {
        return nil;
    }

    ACDataclassAction* cancelAction = [ACDataclassAction actionWithType:ACDataclassActionCancel];
    ACDataclassAction* deleteAction = [ACDataclassAction actionWithType:ACDataclassActionDeleteSyncData];
    ACDataclassAction* keepAction = [ACDataclassAction actionWithType:ACDataclassActionMergeSyncDataIntoLocalData];

    return @[cancelAction, deleteAction, keepAction];
}

- (NSArray*)actionsForDisablingDataclassOnAccount:(ACAccount*)account forDataclass:(NSString*)dataclass
{
    return [self actionsForDeletingAccount:account forDataclass:dataclass];
}


- (BOOL)performAction:(ACDataclassAction*)action forAccount:(ACAccount*)account withChildren:(NSArray*)childAccounts forDataclass:(NSString*)dataclass withError:(NSError**)error
{
    // if the user asked us to delete their data, do that now
    if (action.type == ACDataclassActionDeleteSyncData) {
        CFErrorRef cfLocalError = NULL;
        if (SecDeleteItemsOnSignOut(&cfLocalError)) {
            secnotice("ItemDelete", "Deleted items on sign out");
        } else {
            NSError* localError = CFBridgingRelease(cfLocalError);
            secwarning("ItemDelete: Failed to delete items on sign out: %@", localError);
        }
    }

    return YES;
}

@end
