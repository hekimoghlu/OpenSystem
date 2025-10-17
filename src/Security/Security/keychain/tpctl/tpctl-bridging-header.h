/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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


#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "tpctl-objc.h"
#import <Security/OTConstants.h>

#import "keychain/ot/OTDeviceInformationAdapter.h"
#import "keychain/ot/OTAccountsAdapter.h"
#import "keychain/ot/OTPersonaAdapter.h"

// Needed to interface with IDMS
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#import <AppleAccount/AppleAccount.h>
#import <AppleAccount/AppleAccount_Private.h>
#import <AuthKit/AuthKit.h>
#import <AuthKit/AuthKit_Private.h>
#pragma clang diagnostic pop

#import <AppleAccount/ACAccount+AppleAccount.h>
#import "keychain/ot/proto/generated_source/OTAccountSettings.h"
#import "keychain/ot/proto/generated_source/OTWalrus.h"
#import "keychain/ot/proto/generated_source/OTWebAccess.h"

#import "OSX/utilities/SecInternalReleasePriv.h"
