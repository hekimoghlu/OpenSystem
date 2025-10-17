/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#ifndef SUDOERS_INSULTS_H
#define SUDOERS_INSULTS_H

#if defined(HAL_INSULTS) || defined(GOONS_INSULTS) || defined(CLASSIC_INSULTS) || defined(CSOPS_INSULTS) || defined(PYTHON_INSULTS)

#include "sudo_rand.h"

/*
 * Use one or more set of insults as determined by configure
 */

const char *insults[] = {

# ifdef HAL_INSULTS
#  include "ins_2001.h"
# endif

# ifdef GOONS_INSULTS
#  include "ins_goons.h"
# endif

# ifdef CLASSIC_INSULTS
#  include "ins_classic.h"
# endif

# ifdef CSOPS_INSULTS
#  include "ins_csops.h"
# endif

# ifdef PYTHON_INSULTS
#  include "ins_python.h"
# endif

    NULL

};

/*
 * How may I insult you?  Let me count the ways...
 */
#define NOFINSULTS (nitems(insults) - 1)

/*
 * return a pseudo-random insult.
 */
#define INSULT		(insults[arc4random_uniform(NOFINSULTS)])

#endif /* HAL_INSULTS || GOONS_INSULTS || CLASSIC_INSULTS || CSOPS_INSULTS || PYTHON_INSULTS */

#endif /* SUDOERS_INSULTS_H */
