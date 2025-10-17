/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
// Option class for the CUPS PPD Compiler.
//
// Copyright 2007-2011 by Apple Inc.
// Copyright 2002-2005 by Easy Software Products.
//
// Licensed under Apache License v2.0.  See the file "LICENSE" for more information.
//

//
// Include necessary headers...
//

#include "ppdc-private.h"


//
// 'ppdcOption::ppdcOption()' - Create a new option.
//

ppdcOption::ppdcOption(ppdcOptType    ot,	// I - Option type
                       const char     *n,	// I - Option name
		       const char     *t,	// I - Option text
		       ppdcOptSection s,	// I - Section
                       float          o)	// I - Ordering number
  : ppdcShared()
{
  PPDC_NEW;

  type      = ot;
  name      = new ppdcString(n);
  text      = new ppdcString(t);
  section   = s;
  order     = o;
  choices   = new ppdcArray();
  defchoice = 0;
}


//
// 'ppdcOption::ppdcOption()' - Copy a new option.
//

ppdcOption::ppdcOption(ppdcOption *o)		// I - Template option
{
  PPDC_NEW;

  o->name->retain();
  o->text->retain();
  if (o->defchoice)
    o->defchoice->retain();

  type      = o->type;
  name      = o->name;
  text      = o->text;
  section   = o->section;
  order     = o->order;
  choices   = new ppdcArray(o->choices);
  defchoice = o->defchoice;
}


//
// 'ppdcOption::~ppdcOption()' - Destroy an option.
//

ppdcOption::~ppdcOption()
{
  PPDC_DELETE;

  name->release();
  text->release();
  if (defchoice)
    defchoice->release();
  choices->release();
}


//
// 'ppdcOption::find_choice()' - Find an option choice.
//

ppdcChoice *					// O - Choice or NULL
ppdcOption::find_choice(const char *n)		// I - Name of choice
{
  ppdcChoice	*c;				// Current choice


  for (c = (ppdcChoice *)choices->first(); c; c = (ppdcChoice *)choices->next())
    if (!_cups_strcasecmp(n, c->name->value))
      return (c);

  return (0);
}


//
// 'ppdcOption::set_defchoice()' - Set the default choice.
//

void
ppdcOption::set_defchoice(ppdcChoice *c)	// I - Choice
{
  if (defchoice)
    defchoice->release();

  if (c->name)
    c->name->retain();

  defchoice = c->name;
}
