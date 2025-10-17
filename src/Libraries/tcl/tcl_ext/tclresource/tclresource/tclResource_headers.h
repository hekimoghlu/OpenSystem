/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

// File: "TclResource_headers.h"
//                        Created: 2003-09-22 10:47:15
//              Last modification: 2004-09-01 22:04:56
// Author: Bernard Desgraupes
// Description: Use this header to include the precompiled headers
// on OSX for dylib target built with CW Pro 8


#pragma check_header_flags on

#if __POWERPC__
#include "MW_TclResourceHeaderCarbonX"
#endif

#ifdef MAC_TCL
#undef MAC_TCL
#endif
