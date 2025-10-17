/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
// Contraint class for the CUPS PPD Compiler.
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
// 'ppdcConstraint::ppdcConstraint()' - Create a constraint.
//

ppdcConstraint::ppdcConstraint(const char *o1,	// I - First option
                               const char *c1,	// I - First choice
			       const char *o2,	// I - Second option
			       const char *c2)	// I - Second choice
  : ppdcShared()
{
  PPDC_NEW;

  option1 = new ppdcString(o1);
  choice1 = new ppdcString(c1);
  option2 = new ppdcString(o2);
  choice2 = new ppdcString(c2);
}


//
// 'ppdcConstraint::~ppdcConstraint()' - Destroy a constraint.
//

ppdcConstraint::~ppdcConstraint()
{
  PPDC_DELETE;

  option1->release();
  choice1->release();
  option2->release();
  choice2->release();
}
