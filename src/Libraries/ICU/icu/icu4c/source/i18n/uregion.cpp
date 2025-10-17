/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 16, 2025.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
*****************************************************************************************
* Copyright (C) 2013-2015, International Business Machines Corporation and others.
* All Rights Reserved.
*****************************************************************************************
*/

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING

#include "unicode/uregion.h"
#include "unicode/region.h"
#include "unicode/uenum.h"
#include "unicode/strenum.h"

U_NAMESPACE_USE


U_CAPI const URegion* U_EXPORT2
uregion_getRegionFromCode(const char *regionCode, UErrorCode *status)
{
    return (URegion*)Region::getInstance(regionCode, *status);
}


U_CAPI const URegion* U_EXPORT2
uregion_getRegionFromNumericCode (int32_t code, UErrorCode *status)
{
    return (URegion*)Region::getInstance(code, *status);
}


U_CAPI UEnumeration* U_EXPORT2
uregion_getAvailable(URegionType type, UErrorCode *status)
{
    StringEnumeration* strenum = Region::getAvailable(type, *status);
    return uenum_openFromStringEnumeration( strenum, status );
}


U_CAPI UBool U_EXPORT2
uregion_areEqual(const URegion* uregion, const URegion* otherRegion)
{
    return ( (void*)uregion == (void*)otherRegion );
}


U_CAPI const URegion* U_EXPORT2
uregion_getContainingRegion(const URegion* uregion)
{
    return (URegion*)((Region*)uregion)->getContainingRegion();
}


U_CAPI const URegion* U_EXPORT2
uregion_getContainingRegionOfType(const URegion* uregion, URegionType type)
{
    return (URegion*)((Region*)uregion)->getContainingRegion(type);
}


U_CAPI UEnumeration* U_EXPORT2
uregion_getContainedRegions(const URegion* uregion, UErrorCode *status)
{
    StringEnumeration* strenum = ((Region*)uregion)->getContainedRegions(*status);
    return uenum_openFromStringEnumeration( strenum, status);
}


U_CAPI UEnumeration* U_EXPORT2
uregion_getContainedRegionsOfType(const URegion* uregion, URegionType type, UErrorCode *status)
{
    StringEnumeration* strenum = ((Region*)uregion)->getContainedRegions(type, *status);
    return uenum_openFromStringEnumeration( strenum, status);
}


U_CAPI UBool U_EXPORT2
uregion_contains(const URegion* uregion, const URegion* otherRegion)
{
    return ((Region*)uregion)->contains(*((Region*)otherRegion));
}


U_CAPI UEnumeration* U_EXPORT2
uregion_getPreferredValues(const URegion* uregion, UErrorCode *status)
{
    StringEnumeration* strenum = ((Region*)uregion)->getPreferredValues(*status);
    return uenum_openFromStringEnumeration( strenum, status);
}


U_CAPI const char* U_EXPORT2
uregion_getRegionCode(const URegion* uregion)
{
    return ((Region*)uregion)->getRegionCode();
}


U_CAPI int32_t U_EXPORT2
uregion_getNumericCode(const URegion* uregion)
{
    return ((Region*)uregion)->getNumericCode();
}


U_CAPI URegionType U_EXPORT2
uregion_getType(const URegion* uregion)
{
    return ((Region*)uregion)->getType();
}


#endif /* #if !UCONFIG_NO_FORMATTING */
