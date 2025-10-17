/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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

#include "tasm-options.h"


#ifdef __DEBUG__
#define DEBUG(x) fprintf ## x ;
#else
#define DEBUG(x)
#endif


/* Options Parser */
int
parse_cmdline(int argc, char **argv, opt_option *options, size_t nopts,
              void (*print_error) (const char *fmt, ...))
{
    int errors = 0, warnings = 0;
    size_t i;
    int got_it;

    DEBUG((stderr, "parse_cmdline: entered\n"));

  fail:
    while (--argc) {
        argv++;

        if (argv[0][0] == '/' && argv[0][1]) {        /* opt */
            got_it = 0;
            for (i = 0; i < nopts; i++) {
                char *cmd = &argv[0][1];
                size_t len = strlen(options[i].opt);
                if (yasm__strncasecmp(cmd, options[i].opt, len) == 0) {
                    char *param;

                    param = &argv[0][1+len];
                    if (options[i].takes_param) {
                        if (param[0] == '\0') {
                            print_error(
                                _("option `-%c' needs an argument!"),
                                options[i].opt);
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
            if (!got_it) {
                print_error(_("warning: unrecognized option `%s'"),
                            argv[0]);
                warnings++;
            }
        } else {    /* not an option, then it should be a file or something */

            if (not_an_option_handler(argv[0]))
                errors++;
        }
    }

    DEBUG((stderr, "parse_cmdline: finished\n"));
    return errors;
}

void
help_msg(const char *msg, const char *tail, opt_option *options, size_t nopts)
{
    char optbuf[100];
    size_t i;

    printf("%s", gettext(msg));

    for (i = 0; i < nopts; i++) {
        optbuf[0] = 0;

        if (options[i].takes_param) {
            if (options[i].opt)
                sprintf(optbuf, "/%s <%s>", options[i].opt,
                        options[i].param_desc ? options[i].
                        param_desc : _("param"));
        } else {
            if (options[i].opt)
                sprintf(optbuf, "/%s", options[i].opt);
        }

        printf("    %-22s  %s\n", optbuf, gettext(options[i].description));
    }

    printf("%s", gettext(tail));
}
