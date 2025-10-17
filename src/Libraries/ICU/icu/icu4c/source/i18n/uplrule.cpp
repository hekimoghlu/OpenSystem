/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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
#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/upluralrules.h"
#include "unicode/uplrule.h"

U_NAMESPACE_USE

U_CAPI UPluralRules* U_EXPORT2
uplrule_open(const char *locale,
              UErrorCode *status)
{
    return uplrules_open(locale, status);
}

U_CAPI void U_EXPORT2
uplrule_close(UPluralRules *plrules)
{
    uplrules_close(plrules);
}

U_CAPI int32_t U_EXPORT2
uplrule_select(const UPluralRules *plrules,
               int32_t number,
               UChar *keyword, int32_t capacity,
               UErrorCode *status)
{
    return uplrules_select(plrules, number, keyword, capacity, status);
}

U_CAPI int32_t U_EXPORT2
uplrule_selectDouble(const UPluralRules *plrules,
                     double number,
                     UChar *keyword, int32_t capacity,
                     UErrorCode *status)
{
    return uplrules_select(plrules, number, keyword, capacity, status);
}

#endif /* #if !UCONFIG_NO_FORMATTING */
