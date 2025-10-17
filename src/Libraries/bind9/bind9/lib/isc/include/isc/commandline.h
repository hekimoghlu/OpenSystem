/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
/* $Id: commandline.h,v 1.16 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_COMMANDLINE_H
#define ISC_COMMANDLINE_H 1

/*! \file isc/commandline.h */

#include <isc/boolean.h>
#include <isc/lang.h>
#include <isc/platform.h>

/*% Index into parent argv vector. */
LIBISC_EXTERNAL_DATA extern int isc_commandline_index;
/*% Character checked for validity. */
LIBISC_EXTERNAL_DATA extern int isc_commandline_option;
/*% Argument associated with option. */
LIBISC_EXTERNAL_DATA extern char *isc_commandline_argument;
/*% For printing error messages. */
LIBISC_EXTERNAL_DATA extern char *isc_commandline_progname;
/*% Print error message. */
LIBISC_EXTERNAL_DATA extern isc_boolean_t isc_commandline_errprint;
/*% Reset getopt. */
LIBISC_EXTERNAL_DATA extern isc_boolean_t isc_commandline_reset;

ISC_LANG_BEGINDECLS

/*% parse command line */
int
isc_commandline_parse(int argc, char * const *argv, const char *options);

ISC_LANG_ENDDECLS

#endif /* ISC_COMMANDLINE_H */
