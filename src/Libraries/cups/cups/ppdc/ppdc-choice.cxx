/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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
// Option choice class for the CUPS PPD Compiler.
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
// 'ppdcChoice::ppdcChoice()' - Create a new option choice.
//

ppdcChoice::ppdcChoice(const char *n,	// I - Name of choice
                       const char *t,	// I - Text of choice
		       const char *c)	// I - Code of choice
  : ppdcShared()
{
  PPDC_NEW;

  name = new ppdcString(n);
  text = new ppdcString(t);
  code = new ppdcString(c);
}


//
// 'ppdcChoice::~ppdcChoice()' - Destroy an option choice.
//

ppdcChoice::~ppdcChoice()
{
  PPDC_DELETE;

  name->release();
  text->release();
  code->release();
}
