/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
/*
  * System library.
  */
#include <sys_defs.h>

 /*
  * Utility library.
  */
#include <msg.h>
#include <mymalloc.h>
#include <stringops.h>

 /*
  * Global library.
  */
#include <mail_params.h>
#include <mail_parm_split.h>

/* mail_parm_split - split list, extract {text}, errors are fatal */

ARGV   *mail_parm_split(const char *name, const char *value)
{
    ARGV   *argvp = argv_alloc(1);
    char   *saved_string = mystrdup(value);
    char   *bp = saved_string;
    char   *arg;
    char   *err;

    /*
     * The code that detects the error shall either signal or handle the
     * error. In this case, mystrtokq() detects no error, extpar() signals
     * the error to its caller, and this function handles the error.
     */
    while ((arg = mystrtokq(&bp, CHARS_COMMA_SP, CHARS_BRACE)) != 0) {
	if (*arg == CHARS_BRACE[0]
	    && (err = extpar(&arg, CHARS_BRACE, EXTPAR_FLAG_STRIP)) != 0) {
#ifndef TEST
	    msg_fatal("%s: %s", name, err);
#else
	    msg_warn("%s: %s", name, err);
	    myfree(err);
#endif
	}
	argv_add(argvp, arg, (char *) 0);
    }
    argv_terminate(argvp);
    myfree(saved_string);
    return (argvp);
}

#ifdef TEST

 /*
  * This function is security-critical so it better have a unit-test driver.
  */
#include <string.h>
#include <vstream.h>
#include <vstream.h>
#include <vstring_vstream.h>

int     main(void)
{
    VSTRING *vp = vstring_alloc(100);
    ARGV   *argv;
    char   *start;
    char   *str;
    char  **cpp;

    while (vstring_fgets_nonl(vp, VSTREAM_IN) && VSTRING_LEN(vp) > 0) {
	start = vstring_str(vp);
	vstream_printf("Input:\t>%s<\n", start);
	vstream_fflush(VSTREAM_OUT);
	argv = mail_parm_split("stdin", start);
	for (cpp = argv->argv; (str = *cpp) != 0; cpp++)
	    vstream_printf("Output:\t>%s<\n", str);
	argv_free(argv);
	vstream_fflush(VSTREAM_OUT);
    }
    vstring_free(vp);
    return (0);
}

#endif
