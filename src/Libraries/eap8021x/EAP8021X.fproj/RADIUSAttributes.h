/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
#ifndef _EAP8021X_RADIUSATTRIBUTES_H
#define _EAP8021X_RADIUSATTRIBUTES_H

/*
 * RADIUS.h
 * - definitions for RADIUS attributes
 */

/* 
 * Modification History
 *
 * October 11, 2002	Dieter Siegmund (dieter@apple)
 * - created
 */

#include <stdint.h>

enum {
    kRADIUSAttributeTypeUserName = 1,
    kRADIUSAttributeTypeUserPassword = 2,
    kRADIUSAttributeTypeCHAPPassword = 3,
    kRADIUSAttributeTypeCHAPChallenge = 60,
    kRADIUSAttributeTypeEAPMessage = 79,
};

typedef uint32_t RADIUSAttributeType;

enum {
    kMSRADIUSAttributeTypeMSCHAPResponse = 1,
    kMSRADIUSAttributeTypeMSCHAPError = 2,
    kMSRADIUSAttributeTypeMSCHAPDomain = 10,
    kMSRADIUSAttributeTypeMSCHAPChallenge = 11,
    kMSRADIUSAttributeTypeMSCHAP2Response = 25,
    kMSRADIUSAttributeTypeMSCHAP2Success = 26,
};
typedef uint32_t MSRADIUSAttributeType;

enum {
    kRADIUSVendorIdentifierMicrosoft = 311,
};

typedef uint32_t RADIUSVendorIdentifer;

#endif /* _EAP8021X_RADIUSATTRIBUTES_H */
