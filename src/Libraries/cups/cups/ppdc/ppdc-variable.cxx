/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
// Variable class for the CUPS PPD Compiler.
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
// 'ppdcVariable::ppdcVariable()' - Create a variable.
//

ppdcVariable::ppdcVariable(const char *n,	// I - Name of variable
                           const char *v)	// I - Value of variable
  : ppdcShared()
{
  PPDC_NEW;

  name  = new ppdcString(n);
  value = new ppdcString(v);
}


//
// 'ppdcVariable::~ppdcVariable()' - Destroy a variable.
//

ppdcVariable::~ppdcVariable()
{
  PPDC_DELETE;

  name->release();
  value->release();
}


//
// 'ppdcVariable::set_value()' - Set the value of a variable.
//

void
ppdcVariable::set_value(const char *v)
{
  value->release();
  value = new ppdcString(v);
}
