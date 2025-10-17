/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

#import "utilities/debugging.h"
#import "keychain/keychainupgrader/KeychainItemUpgradeRequestServer.h"
#import "keychain/keychainupgrader/KeychainItemUpgradeRequestServerHelpers.h"

static bool KeychainItemUpgradeRequestServerEnabled = false;
bool KeychainItemUpgradeRequestServerIsEnabled(void) {
    return KeychainItemUpgradeRequestServerEnabled;
}
void KeychainItemUpgradeRequestServerSetEnabled(bool enabled) {
    KeychainItemUpgradeRequestServerEnabled = enabled;
}

void KeychainItemUpgradeRequestServerInitialize(void) {
    secnotice("keychainitemupgrade", "performing KeychainItemUpgradeRequestServerInitialize");
    KeychainItemUpgradeRequestServer* server = [KeychainItemUpgradeRequestServer server];
    
    [server.controller triggerKeychainItemUpdateRPC:^(NSError * _Nullable error) {
        secnotice("keychainitemupgrade", "kicking off keychain item upgrade");
    }];
}
