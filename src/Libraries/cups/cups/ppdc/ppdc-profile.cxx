/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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
// Color profile class for the CUPS PPD Compiler.
//
// Copyright 2007-2009 by Apple Inc.
// Copyright 2002-2005 by Easy Software Products.
//
// Licensed under Apache License v2.0.  See the file "LICENSE" for more information.
//

//
// Include necessary headers...
//

#include "ppdc-private.h"


//
// 'ppdcProfile::ppdcProfile()' - Create a color profile.
//

ppdcProfile::ppdcProfile(const char  *r,	// I - Resolution name
                         const char  *m,	// I - Media type name
			 float       d,		// I - Density
			 float       g,		// I - Gamma
			 const float *p)	// I - 3x3 transform matrix
  : ppdcShared()
{
  PPDC_NEW;

  resolution = new ppdcString(r);
  media_type = new ppdcString(m);
  density    = d;
  gamma      = g;

  memcpy(profile, p, sizeof(profile));
}


//
// 'ppdcProfile::~ppdcProfile()' - Destroy a color profile.
//

ppdcProfile::~ppdcProfile()
{
  PPDC_DELETE;

  resolution->release();
  media_type->release();
}
