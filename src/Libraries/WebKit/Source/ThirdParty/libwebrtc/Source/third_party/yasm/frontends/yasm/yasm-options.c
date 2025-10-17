/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
#include <util.h>
#include <ctype.h>
#include <string.h>
#include "yasm-options.h"


#ifdef __DEBUG__
#define YASM_DEBUG(x) fprintf ## x ;
#else
#define YASM_DEBUG(x)
#endif


/* Options Parser */
int
parse_cmdline(int argc, char **argv, opt_option *options, size_t nopts,
              void (*print_error) (const char *fmt, ...))
{
    int errors = 0, warnings = 0;
    size_t i;
    int got_it;

    YASM_DEBUG((stderr, "parse_cmdline: entered\n"));

  fail:
    while (--argc) {
        argv++;

        if (argv[0][0] == '-') {        /* opt */
            got_it = 0;
            if (argv[0][1] == '-') {    /* lopt */
                if (argv[0][2] == '\0') {   /* --, end of options */
                    /* Handle rest of args as non-options */
                    while (--argc) {
                        argv++;
                        if (not_an_option_handler(argv[0]))
                            errors++;
                    }
                    return errors;
                }

                for (i = 0; i < nopts; i++) {
                    size_t optlen;
                    if (options[i].lopt &&
                        strncmp(&argv[0][2], options[i].lopt,
                                (optlen = strlen(options[i].lopt))) == 0) {
                        char *param;
                        char c = argv[0][2 + optlen];

                        if (c != '\0' && c != '=' && !isspace(c))
                            continue;

                        if (options[i].takes_param) {
                            param = strchr(&argv[0][2], '=');
                            if (!param) {
                                print_error(
                                    _("option `--%s' needs an argument!"),
                                    options[i].lopt);
                                errors++;
                                goto fail;
                            } else {
                                *param = '\0';
                                param++;
                            }
                        } else
                            param = NULL;

                        if (!options[i].
                            handler(&argv[0][2], param, options[i].extra))
                            got_it = 1;
                        break;
                    }
                }
                if (!got_it && !other_option_handler(argv[0]))
                    got_it = 1;
                if (!got_it) {
                    print_error(_("warning: unrecognized option `%s'"),
                                argv[0]);
                    warnings++;
                }
            } else if (argv[0][1] == '\0') {   /* just -, is non-option */
                if (not_an_option_handler(argv[0]))
                    errors++;
            } else {            /* sopt */
                for (i = 0; i < nopts; i++) {
                    if (argv[0][1] == options[i].sopt) {
                        char *cmd = &argv[0][1];
                        char *param;

                        if (options[i].takes_param) {
                            param = argv[1];
                            if (argv[0][2] != '\0')
                                param = &argv[0][2];
                            else if (param == NULL || *param == '-') {
                                print_error(
                                    _("option `-%c' needs an argument!"),
                                    options[i].sopt);
                                errors++;
                                goto fail;
                            } else {
                                argc--;
                                argv++;
                            }
                        } else
                            param = NULL;

                        if (!options[i].handler(cmd, param, options[i].extra))
                            got_it = 1;
                        break;
                    }
                }
                if (!got_it && !other_option_handler(argv[0]))
                    got_it = 1;
                if (!got_it) {
                    print_error(_("warning: unrecognized option `%s'"),
                                argv[0]);
                    warnings++;
                }
            }
        } else {    /* not an option, then it should be a file or something */

            if (not_an_option_handler(argv[0]))
                errors++;
        }
    }

    YASM_DEBUG((stderr, "parse_cmdline: finished\n"));
    return errors;
}

void
help_msg(const char *msg, const char *tail, opt_option *options, size_t nopts)
{
    char optbuf[100], optopt[100];
    size_t i;

    printf("%s", gettext(msg));

    for (i = 0; i < nopts; i++) {
        size_t shortopt_len = 0;
        size_t longopt_len = 0;

        optbuf[0] = 0;
        optopt[0] = 0;

        if (options[i].takes_param) {
            if (options[i].sopt) {
                snprintf(optbuf, 100, "-%c <%s>", options[i].sopt,
                        options[i].param_desc ? options[i].
                        param_desc : _("param"));
                shortopt_len = strlen(optbuf);
            }
            if (options[i].sopt && options[i].lopt)
                strlcat(optbuf, ", ", 100);
            if (options[i].lopt) {
                snprintf(optopt, 100, "--%s=<%s>", options[i].lopt,
                        options[i].param_desc ? options[i].
                        param_desc : _("param"));
                strlcat(optbuf, optopt, 100);
                longopt_len = strlen(optbuf);
            }
        } else {
            if (options[i].sopt) {
                snprintf(optbuf, 100, "-%c", options[i].sopt);
                shortopt_len = strlen(optbuf);
            }
            if (options[i].sopt && options[i].lopt)
                strlcat(optbuf, ", ", 100);
            if (options[i].lopt) {
                snprintf(optopt, 100, "--%s", options[i].lopt);
                strlcat(optbuf, optopt, 100);
                longopt_len = strlen(optbuf);
            }
        }

        /* split [-s <desc>], [--long <desc>] if it destroys columns */
        if (shortopt_len && longopt_len && longopt_len > 22) {
            optbuf[shortopt_len] = '\0';
            printf("    %-22s  %s\n", optopt, gettext(options[i].description));
            printf("     %s\n", optbuf);
        }
        else
            printf("    %-22s  %s\n", optbuf, gettext(options[i].description));
    }

    printf("%s", gettext(tail));
}
