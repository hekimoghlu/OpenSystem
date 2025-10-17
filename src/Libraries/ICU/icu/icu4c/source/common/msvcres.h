/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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
//{{NO_DEPENDENCIES}}
// Copyright (c) 2003-2010 International Business Machines
// Corporation and others. All Rights Reserved.
//
// Used by common.rc and other .rc files.
//Do not edit with Microsoft Developer Studio because it will modify this
//header the wrong way. This is here to prevent Visual Studio .NET from
//unnessarily building the resource files when it's not needed.
//

/*
These are defined before unicode/uversion.h in order to prevent 
STLPort's broken stddef.h from being used when rc.exe parses this file. 
*/
#define _STLP_OUTERMOST_HEADER_ID 0
#define _STLP_WINCE 1

#include "unicode/uversion.h"

#define ICU_WEBSITE "https://icu.unicode.org/"
#define ICU_COMPANY "The ICU Project"
#define ICU_PRODUCT_PREFIX "ICU"
#define ICU_PRODUCT "International Components for Unicode"
