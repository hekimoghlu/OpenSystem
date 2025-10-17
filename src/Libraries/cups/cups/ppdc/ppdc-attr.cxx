/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
// Attribute class for the CUPS PPD Compiler.
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
// 'ppdcAttr::ppdcAttr()' - Create an attribute.
//

ppdcAttr::ppdcAttr(const char *n,	// I - Name
                   const char *s,	// I - Spec string
		   const char *t,	// I - Human-readable text
		   const char *v,	// I - Value
		   bool       loc)	// I - Localize this attribute?
  : ppdcShared()
{
  PPDC_NEW;

  name        = new ppdcString(n);
  selector    = new ppdcString(s);
  text        = new ppdcString(t);
  value       = new ppdcString(v);
  localizable = loc;
}


//
// 'ppdcAttr::~ppdcAttr()' - Destroy an attribute.
//

ppdcAttr::~ppdcAttr()
{
  PPDC_DELETE;

  name->release();
  selector->release();
  text->release();
  value->release();
}
