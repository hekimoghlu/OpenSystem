/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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
 * Cephes Math Library Release 2.0:  April, 1987
 * Copyright 1984, 1987 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

#include <stdio.h>
#include "mconf.h"

#include "sf_error.h"

int merror = 0;

/* Notice: the order of appearance of the following
 * messages is bound to the error codes defined
 * in mconf.h.
 */
static char *ermsg[8] = {
    "unknown",			/* error code 0 */
    "domain",			/* error code 1 */
    "singularity",		/* et seq.      */
    "overflow",
    "underflow",
    "total loss of precision",
    "partial loss of precision",
    "too many iterations"
};

static sf_error_t conv_to_sf[8] = {
    SF_ERROR_OTHER,
    SF_ERROR_DOMAIN,
    SF_ERROR_SINGULAR,
    SF_ERROR_OVERFLOW,
    SF_ERROR_UNDERFLOW,
    SF_ERROR_NO_RESULT,
    SF_ERROR_LOSS,
    SF_ERROR_SLOW
};

int mtherr(char *name, int code)
{
    /* Display string passed by calling program,
     * which is supposed to be the name of the
     * function in which the error occurred:
     */

    /* Set global error message word */
    merror = code;

    /* Display error message defined
     * by the code argument.
     */
    if ((code <= 0) || (code >= 8))
	code = 0;

    sf_error(name, conv_to_sf[code], NULL);

    /* Return to calling
     * program
     */
    return (0);
}
