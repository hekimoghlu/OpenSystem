/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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
*******************************************************************************
* Copyright (C) 2013, International Business Machines Corporation and         *
* others. All Rights Reserved.                                                *
*******************************************************************************
*
* File REGION_IMPL.H
*
*******************************************************************************
*/

#ifndef __REGION_IMPL_H__
#define __REGION_IMPL_H__

#include "unicode/utypes.h"

#if !UCONFIG_NO_FORMATTING
    
#include "uvector.h"
#include "unicode/strenum.h"

U_NAMESPACE_BEGIN


class RegionNameEnumeration : public StringEnumeration {
public:
    /**
     * Construct an string enumeration over the supplied name list.
     * Makes a copy of the supplied input name list; does not retain a reference to the original.
     */
    RegionNameEnumeration(UVector *nameList, UErrorCode& status);
    virtual ~RegionNameEnumeration();
    static UClassID U_EXPORT2 getStaticClassID();
    virtual UClassID getDynamicClassID() const override;
    virtual const UnicodeString* snext(UErrorCode& status) override;
    virtual void reset(UErrorCode& status) override;
    virtual int32_t count(UErrorCode& status) const override;
private:
    int32_t pos;
    UVector *fRegionNames;
};

U_NAMESPACE_END

#endif

#endif
