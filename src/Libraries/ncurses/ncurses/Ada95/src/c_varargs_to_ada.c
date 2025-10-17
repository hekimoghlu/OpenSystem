/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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
/****************************************************************************
 *   Author:  Nicolas Boulenguez, 2011                                      *
 ****************************************************************************/

/*
    Version Control
    $Id: c_varargs_to_ada.c,v 1.6 2014/05/24 21:32:18 tom Exp $
  --------------------------------------------------------------------------*/
/*
  */

#include "c_varargs_to_ada.h"

int
set_field_type_alnum(FIELD *field,
		     int minimum_width)
{
  return set_field_type(field, TYPE_ALNUM, minimum_width);
}

int
set_field_type_alpha(FIELD *field,
		     int minimum_width)
{
  return set_field_type(field, TYPE_ALPHA, minimum_width);
}

int
set_field_type_enum(FIELD *field,
		    char **value_list,
		    int case_sensitive,
		    int unique_match)
{
  return set_field_type(field, TYPE_ENUM, value_list, case_sensitive,
			unique_match);
}

int
set_field_type_integer(FIELD *field,
		       int precision,
		       long minimum,
		       long maximum)
{
  return set_field_type(field, TYPE_INTEGER, precision, minimum, maximum);
}

int
set_field_type_numeric(FIELD *field,
		       int precision,
		       double minimum,
		       double maximum)
{
  return set_field_type(field, TYPE_NUMERIC, precision, minimum, maximum);
}

int
set_field_type_regexp(FIELD *field,
		      char *regular_expression)
{
  return set_field_type(field, TYPE_REGEXP, regular_expression);
}

int
set_field_type_ipv4(FIELD *field)
{
  return set_field_type(field, TYPE_IPV4);
}

int
set_field_type_user(FIELD *field,
		    FIELDTYPE *fieldtype,
		    void *arg)
{
  return set_field_type(field, fieldtype, arg);
}

void *
void_star_make_arg(va_list *list)
{
  return va_arg(*list, void *);
}

#ifdef TRACE
void
_traces(const char *fmt, char *arg)
{
  _tracef(fmt, arg);
}
#endif
