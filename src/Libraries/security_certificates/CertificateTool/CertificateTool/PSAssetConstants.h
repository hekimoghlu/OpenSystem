/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
//  PSAssetConstants.h
//  CertificateTool
//
//  Copyright (c) 2013-2015,2024 Apple Inc. All Rights Reserved.
//

#ifndef _PSAssetConstants_h
#define _PSAssetConstants_h

#include <CoreFoundation/CoreFoundation.h>

enum
{
    isAnchor = (1UL << 0),
    isBlocked = (1UL << 1),
    isGrayListed = (1UL << 2),
    hasFullCert = (1UL << 3),
    hasCertHash = (1UL << 4),
    isAllowListed = (1UL << 5),
    isSystem = (1UL << 6),
    isPlatform = (1UL << 7),
    isCustom = (1UL << 8),
};

typedef unsigned long PSAssetFlags;

extern const CFStringRef kPSAssertCertificatesKey;
extern const CFStringRef kPSAssertVersionNumberKey;
extern const CFStringRef kPSAssetCertDataKey;
extern const CFStringRef kPSAssetCertHashKey;
extern const CFStringRef kPSAssetCertEVOIDSKey;
extern const CFStringRef kPSAssetCertFlagsKey;
extern const CFStringRef kPSAssertAdditionalDataKey;


#endif
