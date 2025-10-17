/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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

//
//  SOSRingPeerInfoUtils.c
//  sec
//
//  Created by Richard Murphy on 3/15/15.
//
//

#include "SOSRingPeerInfoUtils.h"
#include <AssertMacros.h>

#include "keychain/SecureObjectSync/SOSInternal.h"
#include "keychain/SecureObjectSync/SOSPeer.h"
#include "keychain/SecureObjectSync/SOSPeerInfoInternal.h"
#include "keychain/SecureObjectSync/SOSPeerInfoCollections.h"
#include "keychain/SecureObjectSync/SOSCircle.h"
#include <Security/SecFramework.h>

#include <Security/SecKey.h>
#include <Security/SecKeyPriv.h>
#include <CoreFoundation/CoreFoundation.h>

#include <utilities/SecCFWrappers.h>

//#include "ckdUtilities.h"

#include <corecrypto/ccder.h>
#include <corecrypto/ccdigest.h>
#include <corecrypto/ccsha2.h>


#include <utilities/der_plist.h>
#include <utilities/der_plist_internal.h>
#include <corecrypto/ccder.h>
#include <utilities/der_date.h>

#include <stdlib.h>

#include "SOSRing.h"
#include "SOSRingUtils.h"
