/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
//  SOSAccountGhost.h
//  sec
//
//
//

#ifndef SOSAccountGhost_h
#define SOSAccountGhost_h

#include "SOSAccount.h"
#include "keychain/SecureObjectSync/SOSTypes.h"

#define GHOSTBUST_PERIODIC 1

bool SOSAccountGhostResultsInReset(SOSAccount* account);
CF_RETURNS_RETAINED SOSCircleRef SOSAccountCloneCircleWithoutMyGhosts(SOSAccount* account, SOSCircleRef startCircle);

#if __OBJC__
@class SOSAuthKitHelpers;

/*
 * Ghostbust devices that are not in circle
 *
 * @param account account to operate on
 * @param akh injection of parameters
 * @param mincount if circle is smaller the
 *
 * @return true if there was a device busted
 */

bool SOSAccountGhostBustCircle(SOSAccount *account, SOSAuthKitHelpers *akh, SOSAccountGhostBustingOptions options, int mincount);

#endif

#endif /* SOSAccountGhost_h */
