/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
**********************************************************************
* Copyright (c) 2003-2013, International Business Machines
* Corporation and others.  All Rights Reserved.
**********************************************************************
* Author: Alan Liu
* Created: July 10 2003
* Since: ICU 2.8
**********************************************************************
*/

#ifndef _TZ2ICU_H_
#define _TZ2ICU_H_

/* We have modified the zoneinfo binary format (we write raw offset
 * and DST offset separately instead of their sum) so we notate the
 * file with a distinct signature.  This prevents someone from trying
 * to use our output files as normal zoneinfo files, and also prevents
 * someone from trying to use normal zoneinfo files for ICU.  We also
 * use the first byte of the reserved section as a version integer, to
 * be incremented each time the data format changes.
 */

#define TZ_ICU_MAGIC "TZic" /* cf. TZ_MAGIC = "TZif" */

typedef unsigned char ICUZoneinfoVersion;

#define TZ_ICU_VERSION ((ICUZoneinfoVersion) 1)

/* File into which we will write supplemental ICU data.  This allows
 * zic to communicate final zone data to tz2icu. */
#define ICU_ZONE_FILE "icu_zone.txt"

/* Output resource name.  This determines both the file name and the
 * resource name within the file.  That is, the output will be to the
 * file ICU_TZ_RESOURCE ".txt" and the resource within it will be
 * ICU_TZ_RESOURCE. */
#define ICU_TZ_RESOURCE_OLD "zoneinfo"
#define ICU_TZ_RESOURCE "zoneinfo64"

/* File containing custom zone-region mapping. */
#define ICU_REGIONS "icuregions"

#endif
