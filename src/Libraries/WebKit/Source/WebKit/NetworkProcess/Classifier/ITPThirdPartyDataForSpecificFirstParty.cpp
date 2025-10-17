/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
#include "config.h"
#include "ITPThirdPartyDataForSpecificFirstParty.h"

#include <wtf/text/MakeString.h>

namespace WebKit {

String ITPThirdPartyDataForSpecificFirstParty::toString() const
{
    return makeString("Has been granted storage access under "_s, firstPartyDomain.string(), ": "_s, storageAccessGranted ? '1' : '0', "; Has been seen under "_s, firstPartyDomain.string(), " in the last 24 hours: "_s, WallTime::now().secondsSinceEpoch() - timeLastUpdated < 24_h ? '1' : '0');
}

bool ITPThirdPartyDataForSpecificFirstParty::operator==(const ITPThirdPartyDataForSpecificFirstParty& other) const
{
    return firstPartyDomain == other.firstPartyDomain && storageAccessGranted == other.storageAccessGranted;
}

} // namespace WebKit
