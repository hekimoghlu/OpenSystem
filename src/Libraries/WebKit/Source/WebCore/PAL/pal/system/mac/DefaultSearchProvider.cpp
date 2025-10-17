/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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
#include "DefaultSearchProvider.h"

#if PLATFORM(MAC)

#include "CFUtilitiesSPI.h"
#include <CoreFoundation/CoreFoundation.h>
#include <wtf/cf/TypeCastsCF.h>

namespace PAL {

RetainPtr<CFStringRef> defaultSearchProviderDisplayName()
{
    auto providerInfo = adoptCF(_CFWebServicesCopyProviderInfo(kCFWebServicesTypeWebSearch, nullptr));
    if (!providerInfo)
        return CFSTR("Google");

    auto displayName = dynamic_cf_cast<CFStringRef>(CFDictionaryGetValue(providerInfo.get(), kCFWebServicesProviderDefaultDisplayNameKey));
    if (!displayName)
        return CFSTR("Google");

    return displayName;
}

} // namespace PAL

#endif // PLATFORM(MAC)
