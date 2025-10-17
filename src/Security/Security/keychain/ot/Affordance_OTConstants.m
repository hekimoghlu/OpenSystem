/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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
#import "keychain/ot/Affordance_OTConstants.h"
#import "utilities/debugging.h"

// SecKeychainStaticPersistentRefs feature flag (for checking if static persistent refs are enabled) is enabled by defualt.The following utilities help in existing tests to set/unset this ff.
// Track the last override value, to suppress logging if it hasnt changed
static bool SecKeychainStaticPersistentRefsEnabledOverrideSet = false;
static bool SecKeychainStaticPersistentRefsEnabledOverride = false;
static BOOL persistentRefOverrideLastValue = false;
bool SecKeychainIsStaticPersistentRefsEnabled(void)
{
    if(SecKeychainStaticPersistentRefsEnabledOverrideSet) {

        if(persistentRefOverrideLastValue != SecKeychainStaticPersistentRefsEnabledOverride) {
            secnotice("octagon", "Static Persistent Refs are %@ (overridden)", SecKeychainStaticPersistentRefsEnabledOverride ? @"enabled" : @"disabled");
            persistentRefOverrideLastValue = SecKeychainStaticPersistentRefsEnabledOverride;
        }
        return SecKeychainStaticPersistentRefsEnabledOverride;
    }

    // SecKeychainStaticPersistentRefs ff is default enabled
    return true;
}

void SecKeychainSetOverrideStaticPersistentRefsIsEnabled(bool value)
{
    SecKeychainStaticPersistentRefsEnabledOverrideSet = true;
    SecKeychainStaticPersistentRefsEnabledOverride = value;
}
