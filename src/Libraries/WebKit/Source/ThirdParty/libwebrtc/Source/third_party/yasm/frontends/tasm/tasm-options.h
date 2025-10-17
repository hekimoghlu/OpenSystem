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
#ifndef TASM_OPTIONS_H
#define TASM_OPTIONS_H

/* an option structure
 * operate on either -sopt, --lopt, -sopt <val> or --lopt=<val>
 */
typedef struct opt_option_s
{
    /* option */
    const char *opt;

    /* !=0 if option requires parameter, 0 if not */
    int takes_param;

    int (*handler) (char *cmd, /*@null@*/ char *param, int extra);
    int extra;                 /* extra value for handler */

    /* description to use in help_msg() */
    /*@observer@*/ const char *description;

    /* optional description for the param taken (NULL if not present) */
    /*  (short - will be printed after option sopt/lopt) */
    /*@observer@*/ /*@null@*/ const char *param_desc;
} opt_option;

/* handle everything that is not an option */
int not_an_option_handler(char *param);

/* parse command line calling handlers when appropriate
 * argc, argv - pass directly from main(argc,argv)
 * options - array of options
 * nopts - options count
 */
int parse_cmdline(int argc, char **argv, opt_option *options, size_t nopts,
                  void (*print_error) (const char *fmt, ...));

/* display help message msg followed by list of options in options and followed
 * by tail
 */
void help_msg(const char *msg, const char *tail, opt_option *options,
              size_t nopts);

#endif
