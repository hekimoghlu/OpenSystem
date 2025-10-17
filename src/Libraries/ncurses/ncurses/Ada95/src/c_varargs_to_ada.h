/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
/* $Id: c_varargs_to_ada.h,v 1.4 2015/08/06 23:08:47 tom Exp $ */

#ifndef __C_VARARGS_TO_ADA_H
#define __C_VARARGS_TO_ADA_H

#ifdef HAVE_CONFIG_H
#include <ncurses_cfg.h>
#else
#include <ncurses.h>
#endif

#include <stdlib.h>

#include <form.h>

extern int set_field_type_alnum(FIELD * /* field */ ,
				int /* minimum_width */ );

extern int set_field_type_alpha(FIELD * /* field */ ,
				int /* minimum_width */ );

extern int set_field_type_enum(FIELD * /* field */ ,
			       char ** /* value_list */ ,
			       int /* case_sensitive */ ,
			       int /* unique_match */ );

extern int set_field_type_integer(FIELD * /* field */ ,
				  int /* precision */ ,
				  long /* minimum */ ,
				  long /* maximum */ );

extern int set_field_type_numeric(FIELD * /* field */ ,
				  int /* precision */ ,
				  double /* minimum */ ,
				  double /* maximum */ );

extern int set_field_type_regexp(FIELD * /* field */ ,
				 char * /* regular_expression */ );

extern int set_field_type_ipv4(FIELD * /* field */ );

extern int set_field_type_user(FIELD * /* field */ ,
			       FIELDTYPE * /* fieldtype */ ,
			       void * /* arg */ );

extern void *void_star_make_arg(va_list * /* list */ );

#ifdef TRACE
extern void _traces(const char *	/* fmt */
		    ,char * /* arg */ );
#endif

#endif /* __C_VARARGS_TO_ADA_H */
